"""MCTS search strategy for Kartezio (current master API, Aug 2026).

Design goal
-----------
Keep Kartezio's existing initializer, MutationHandler, mutation behaviors
(including AccumulateBehavior), DecoderCGP and Fitness unchanged.

MCTS owns a *whole evolutionary step* because tree search needs to interleave:
    mutate -> decode/evaluate -> tree decision -> backpropagate

The current Kartezio Strategy API separates reproduction from evaluation, so
this class adds ``step(...)``. A tiny hook in ``KartezioCGP.evolve`` is needed;
see the patch at the bottom of this file.

Fitness convention
------------------
Kartezio minimizes fitness (0 is optimal). MCTS internally maximizes reward,
so reward = -fitness.
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Literal

import numpy as np

from kartezio.evolution.population import PopulationHistory, PopulationWithElite
from kartezio.evolution.strategy import Strategy
from kartezio.mutation.handler import MutationHandler


@dataclass(slots=True)
class MCTSConfig:
    # Number of *new, uncached* genotype evaluations per outer generation.
    evaluation_budget: int = 100

    # Maximum number of children explicitly opened from one tree node.
    branching_factor: int = 8

    # Maximum number of successive Kartezio mutations explored below the elite.
    max_depth: int = 3

    # UCT exploration constant. sqrt(2) is the conventional starting point.
    exploration: float = math.sqrt(2.0)

    # Number of genomes returned to PopulationWithElite as the lambda children.
    # If None, uses strategy.n_children.
    n_return: int | None = None

    # Retry when mutation recreates a genotype already present under that node.
    duplicate_retries: int = 8

    # Standard MCTS uses mean backup. "max" is often interesting for pure
    # optimization because a branch is valuable if it contains one great child.
    backup: Literal["mean", "max"] = "mean"

    # Keep evaluation results by exact genotype encoding during one fit/search.
    use_cache: bool = True

    # Optional cap on total tree nodes. None = no explicit cap.
    max_tree_nodes: int | None = None


@dataclass(slots=True)
class Evaluation:
    fitness: float
    raw: np.ndarray
    elapsed: float


class _Node:
    __slots__ = (
        "genotype",
        "key",
        "parent",
        "depth",
        "children",
        "visits",
        "value_sum",
        "value_max",
        "evaluation",
    )

    def __init__(self, genotype, key: bytes, parent=None, depth: int = 0):
        self.genotype = genotype
        self.key = key
        self.parent = parent
        self.depth = depth
        self.children: list[_Node] = []
        self.visits = 0
        self.value_sum = 0.0
        self.value_max = -math.inf
        self.evaluation: Evaluation | None = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS(Strategy):
    """Drop-in-ish Kartezio strategy with one additional ``step`` hook.

    The class intentionally keeps ``initializer`` and ``mutation_handler`` as
    public attributes because current Kartezio expects them on its Strategy.
    """

    def __init__(
        self,
        initializer,
        mutation_handler: MutationHandler,
        config: MCTSConfig | None = None,
    ):
        self.n_parents = 1
        self.n_children = 4
        self.initializer = initializer
        self.mutation_handler = mutation_handler
        self.config = config or MCTSConfig()

        # Preserve OnePlusLambda knobs / trainer compatibility.
        self.gamma = None
        self.required_fps = None

        # Search-local state.
        self._cache: dict[bytes, Evaluation] = {}
        self._decoder = None
        self._fitness = None
        self._x = None
        self._y = None

        # Diagnostics exposed after each generation.
        self.last_new_evaluations = 0
        self.last_cache_hits = 0
        self.last_tree_nodes = 0
        self.last_max_depth = 0

    @classmethod
    def from_strategy(
        cls, strategy: Strategy, config: MCTSConfig | None = None
    ) -> "MCTS":
        """Build MCTS from Kartezio's existing OnePlusLambda strategy.

        Example:
            trainer.model.evolver.strategy = MCTS.from_strategy(
                trainer.strategy,
                MCTSConfig(evaluation_budget=100, branching_factor=8, max_depth=3),
            )
        """
        obj = cls(strategy.initializer, strategy.mutation_handler, config=config)
        obj.n_children = getattr(strategy, "n_children", 4)
        obj.gamma = getattr(strategy, "gamma", None)
        obj.required_fps = getattr(strategy, "required_fps", None)
        return obj

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_required_fps(self, required_fps):
        self.required_fps = 1.0 / required_fps

    def clear_cache(self):
        self._cache.clear()

    def compile(self, n_iterations: int) -> PopulationWithElite:
        self.mutation_handler.compile(n_iterations)
        self.clear_cache()
        return self.create_population()

    def create_population(self) -> PopulationWithElite:
        population = PopulationWithElite(self.n_children)
        for i in range(population.size):
            population.individuals[i] = self.initializer.random()
        return population

    # Kept for Strategy compatibility. In MCTS mode evolve() should call step().
    def reproduction(self, population: PopulationWithElite):
        raise RuntimeError(
            "MCTS needs interleaved mutation/evaluation. "
            "Patch KartezioCGP.evolve() to call strategy.step(...); "
            "see the integration snippet at the bottom of mcts.py."
        )

    def selection(self, population: PopulationWithElite) -> PopulationHistory:
        """Same selection semantics as current OnePlusLambda."""
        fitness = population.get_raw()

        if self.gamma is not None:
            noise_shape = fitness[0].shape
            noise = np.random.normal(1.0, self.gamma, noise_shape)
            fitness = fitness * noise

        population.score.fitness = np.mean(fitness, axis=1)

        if self.required_fps:
            for i in range(population.size):
                if population.score.time[i] > self.required_fps:
                    population.score.fitness[i] = np.inf

        return population.promote_new_parent()

    # ------------------------------------------------------------------
    # Public MCTS step
    # ------------------------------------------------------------------
    def step(self, population, x, y, decoder, fitness) -> PopulationHistory:
        """Run one complete MCTS-powered Kartezio generation.

        ``population[0]`` is the current elite. Tree search spends
        ``evaluation_budget`` *new* evaluations below it, then writes the best
        discovered genomes into population children and performs normal
        Kartezio elitist selection.
        """
        self._decoder = decoder
        self._fitness = fitness
        self._x = x
        self._y = y

        self.last_new_evaluations = 0
        self.last_cache_hits = 0
        self.last_tree_nodes = 1
        self.last_max_depth = 0

        elite = population.get_elite().clone()
        root_key = self.genotype_key(elite)
        root = _Node(elite, root_key, parent=None, depth=0)

        # Reuse elite evaluation already present in PopulationWithElite.
        if population.score.raw is not None:
            root_eval = Evaluation(
                fitness=float(population.score.fitness[0]),
                raw=np.asarray(population.score.raw[0]).copy(),
                elapsed=float(population.score.time[0]),
            )
            root.evaluation = root_eval
            if self.config.use_cache:
                self._cache[root_key] = root_eval
            self._backup(root, -root_eval.fitness)
        else:
            root.evaluation = self._evaluate(elite)
            self._backup(root, -root.evaluation.fitness)

        discovered: dict[bytes, _Node] = {root_key: root}

        # Budget means actual expensive decoder+fitness calls, not cache hits.
        # A separate guard prevents pathological duplicate-only loops.
        attempts = 0
        max_attempts = max(self.config.evaluation_budget * 20, 100)

        while (
            self.last_new_evaluations < self.config.evaluation_budget
            and attempts < max_attempts
        ):
            attempts += 1

            if (
                self.config.max_tree_nodes is not None
                and self.last_tree_nodes >= self.config.max_tree_nodes
            ):
                break

            node = self._select_for_expansion(root)
            if node is None:
                break

            child = self._expand(node)
            if child is None:
                # Could not obtain a distinct mutant after retries.
                continue

            self.last_tree_nodes += 1
            self.last_max_depth = max(self.last_max_depth, child.depth)
            discovered.setdefault(child.key, child)

            child.evaluation = self._evaluate(child.genotype)
            reward = -child.evaluation.fitness
            self._backup(child, reward)

        # Return the best distinct genomes found anywhere in the tree.
        candidates = [
            n
            for k, n in discovered.items()
            if k != root_key and n.evaluation is not None
        ]
        candidates.sort(key=lambda n: n.evaluation.fitness)

        n_return = self.config.n_return or self.n_children
        selected = candidates[:n_return]

        # Degenerate case: if the search could not produce enough distinct
        # children, fill the remaining population slots with elite clones.
        # This does not exceed the configured evaluation budget.
        while len(selected) < self.n_children:
            node = _Node(elite.clone(), root_key, parent=root, depth=0)
            node.evaluation = Evaluation(
                root.evaluation.fitness,
                root.evaluation.raw.copy(),
                root.evaluation.elapsed,
            )
            selected.append(node)

        selected = selected[: self.n_children]
        self._write_children(population, selected)
        return self.selection(population)

    # ------------------------------------------------------------------
    # Tree search internals
    # ------------------------------------------------------------------
    def _select_for_expansion(self, root: _Node) -> _Node | None:
        node = root

        while True:
            if node.depth >= self.config.max_depth:
                return None

            if len(node.children) < self.config.branching_factor:
                return node

            # Ignore subtrees that are already completely expanded down to
            # max_depth; otherwise UCT can keep selecting a terminal branch.
            candidates = [c for c in node.children if not self._is_saturated(c)]
            if not candidates:
                return None

            node = max(candidates, key=lambda c: self._uct(node, c))

    def _is_saturated(self, node: _Node) -> bool:
        if node.depth >= self.config.max_depth:
            return True
        if len(node.children) < self.config.branching_factor:
            return False
        return all(self._is_saturated(child) for child in node.children)

    def _uct(self, parent: _Node, child: _Node) -> float:
        if child.visits == 0:
            return math.inf

        exploitation = (
            child.mean_value if self.config.backup == "mean" else child.value_max
        )
        exploration = self.config.exploration * math.sqrt(
            math.log(max(parent.visits, 1)) / child.visits
        )
        return exploitation + exploration

    def _expand(self, parent: _Node) -> _Node | None:
        sibling_keys = {c.key for c in parent.children}

        for _ in range(self.config.duplicate_retries + 1):
            genotype = self.mutation_handler.mutate(parent.genotype.clone())
            key = self.genotype_key(genotype)
            if key in sibling_keys:
                continue
            child = _Node(genotype, key, parent=parent, depth=parent.depth + 1)
            parent.children.append(child)
            return child

        return None

    def _backup(self, node: _Node, reward: float):
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node.value_max = max(node.value_max, reward)
            node = node.parent

    # ------------------------------------------------------------------
    # Evaluation + cache
    # ------------------------------------------------------------------
    def _evaluate(self, genotype) -> Evaluation:
        key = self.genotype_key(genotype)

        if self.config.use_cache and key in self._cache:
            self.last_cache_hits += 1
            cached = self._cache[key]
            return Evaluation(cached.fitness, cached.raw.copy(), cached.elapsed)

        y_pred, elapsed = self._decoder.decode(genotype, self._x)
        raw = self._fitness.batch(self._y, [y_pred], reduction="raw")[0]
        raw = np.asarray(raw, dtype=np.float32)
        value = float(np.mean(raw))

        result = Evaluation(value, raw, float(elapsed))
        self.last_new_evaluations += 1

        if self.config.use_cache:
            self._cache[key] = Evaluation(value, raw.copy(), float(elapsed))

        return result

    @staticmethod
    def genotype_key(genotype) -> bytes:
        """Stable exact-DNA hash for current Kartezio Genotype/Chromosome.

        Includes chromosome names, sequence names, dtype, shape and bytes.
        This deliberately hashes the complete genotype, including inactive DNA.
        A future phenotype cache can instead hash decoder.parse_to_graphs(...) +
        active genes if desired.
        """
        h = hashlib.blake2b(digest_size=20)

        for chromosome_name in sorted(genotype._chromosomes.keys()):
            h.update(chromosome_name.encode("utf8"))
            chromosome = genotype._chromosomes[chromosome_name]

            for sequence_name in sorted(chromosome.sequence.keys()):
                arr = np.ascontiguousarray(chromosome.sequence[sequence_name])
                h.update(sequence_name.encode("utf8"))
                h.update(arr.dtype.str.encode("ascii"))
                h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
                h.update(arr.tobytes())

        return h.digest()

    def _write_children(self, population: PopulationWithElite, nodes: list[_Node]):
        if population.score.raw is None:
            n_samples = len(nodes[0].evaluation.raw)
            population.score.raw = np.full(
                (population.size, n_samples), np.inf, dtype=np.float32
            )

        for i, node in enumerate(nodes, start=1):
            population.individuals[i] = node.genotype.clone()
            population.score.raw[i] = node.evaluation.raw
            population.score.fitness[i] = node.evaluation.fitness
            population.score.time[i] = node.evaluation.elapsed


# ---------------------------------------------------------------------------
# MINIMAL KARTEZIO INTEGRATION PATCH
# ---------------------------------------------------------------------------
#
# 1) In KartezioCGP.evolve(), replace the three lines inside the while loop:
#
#       self.evolver.reproduction()
#       self.evaluation(x, y)
#       state = self.evolver.selection()
#
#    with:
#
#       strategy = self.evolver.strategy
#       if hasattr(strategy, "step"):
#           state = strategy.step(
#               self.population,
#               x,
#               y,
#               self.decoder,
#               self.evolver.fitness,
#           )
#       else:
#           self.evolver.reproduction()
#           self.evaluation(x, y)
#           state = self.evolver.selection()
#
# 2) Optional clean public setter in KartezioTrainer:
#
#       def set_strategy(self, strategy: Strategy):
#           self.model.evolver.strategy = strategy
#
# 3) Usage:
#
#       from kartezio.evolution.mcts import MCTS, MCTSConfig
#
#       model = KartezioTrainer(...)
#       model.set_mutation_rates(node_rate=0.05, out_rate=0.1)
#       model.set_behavior(AccumulateBehavior())
#
#       model.set_strategy(
#           MCTS.from_strategy(
#               model.strategy,
#               MCTSConfig(
#                   evaluation_budget=100,
#                   branching_factor=8,
#                   max_depth=3,
#                   exploration=1.414,
#                   backup="mean",
#               ),
#           )
#       )
#
#       elite, history = model.fit(n_iterations, train_x, train_y)
#
# NOTE ON BUDGET:
# Current OnePlusLambda with n_children=4 evaluates 4 new children per generation
# (the elite is not decoded again by decode_population). Therefore 100 MCTS new
# evaluations/generation is 25x the evaluation budget per generation. To keep
# the same approximate total number of *new child evaluations*, 1000 standard
# generations correspond to about 40 MCTS generations, not 250.
