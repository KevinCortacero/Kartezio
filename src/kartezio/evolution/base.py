from abc import ABC, abstractmethod
from typing import Any, List

from kartezio.callback import Callback, Event
from kartezio.core.components import (
    Endpoint,
    Library,
    Preprocessing,
    UpdatableComponent,
)
from kartezio.evolution.decoder import Adapter, DecoderCGP
from kartezio.evolution.fitness import Fitness
from kartezio.evolution.strategy import OnePlusLambda
from kartezio.export import PythonClassWriter
from kartezio.helpers import Observable
from kartezio.mutation.behavioral import MutationBehavior
from kartezio.mutation.decay import MutationDecay
from kartezio.mutation.handler import MutationHandler


class ObservableModel(Observable):
    def send_event(self, name, state):
        self.notify(Event(self.get_current_iteration(), name, state))

    def force_event(self, name, state):
        self.notify(
            Event(self.get_current_iteration(), name, state, force=True)
        )

    def get_current_iteration(self):
        pass


class GeneticAlgorithm:
    def __init__(self, adapter: Adapter, fitness: Fitness):
        """Initialize the genetic algorithm."""
        self.strategy = OnePlusLambda(adapter)
        self.population = None
        self.fitness = fitness
        self.current_iteration = 0
        self.n_iterations = 0

    def initialize(self, n_iterations: int) -> None:
        """Compile the genetic algorithm with given iterations and children."""
        self.population = self.strategy.compile(n_iterations)
        self.current_iteration = 0
        self.n_iterations = n_iterations

    def is_satisfying(self):
        return (
            self.current_iteration >= self.n_iterations
            or self.population.get_fitness()[0] == 0.0
        )

    def selection(self):
        return self.strategy.selection(self.population)

    def reproduction(self):
        self.strategy.reproduction(self.population)

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.batch(y_true, y_pred, reduction="raw")
        self.population.set_raw_fitness(fitness)

    def next(self):
        self.current_iteration += 1


class KartezioCGP(ObservableModel):
    def __init__(
        self,
        n_inputs,
        n_nodes,
        libraries,
        endpoint,
        fitness,
        preprocessing=None,
    ):
        super().__init__()
        self.preprocessing = preprocessing
        self.decoder = DecoderCGP(n_inputs, n_nodes, libraries, endpoint)
        self.evolver = GeneticAlgorithm(self.decoder.adapter, fitness)

    def collect_updatables(self):
        updatables = []
        updatables.extend(
            self.evolver.strategy.mutation_handler.collect_updatables()
        )
        return updatables

    def initialize(self, n_iterations):
        self.evolver.initialize(n_iterations)
        self.clear()
        for updatable in self.collect_updatables():
            self.attach(updatable)

    def get_current_iteration(self):
        return self.evolver.current_iteration

    @property
    def population(self):
        return self.evolver.population

    @property
    def elite(self):
        return self.population.get_elite()

    def evaluation(self, x: List[Any], y: List[Any]):
        y_pred = self.decoder.decode_population(self.population, x)
        self.evolver.evaluation(y, y_pred)

    def evolve(self, x: List[Any], y: List[Any]):
        self.evaluation(x, y)
        changed, state = self.evolver.selection()
        if changed:
            self.force_event(Event.Events.NEW_PARENT, state)
        self.force_event(Event.Events.START_LOOP, state)
        while not self.evolver.is_satisfying():
            self.send_event(Event.Events.START_STEP, state)
            self.evolver.reproduction()
            self.evaluation(x, y)
            changed, state = self.evolver.selection()
            if changed:
                self.force_event(Event.Events.NEW_PARENT, state)
            self.send_event(Event.Events.END_STEP, state)
            self.evolver.next()
        self.force_event(Event.Events.END_LOOP, state)
        return self.elite, state

    def preprocess(self, x):
        if self.preprocessing:
            return self.preprocessing.call(x)
        return x

    def predict(self, x):
        """
        Predict the output of the model given the input.
        Apply the preprocessing if it exists.
        """
        x = self.preprocess(x)
        return self.decoder.decode(self.elite, x)

    def evaluate(self, x, y):
        y_pred, _ = self.predict(x)
        return self.evolver.fitness.batch(y, [y_pred])


class KartezioTrainer:
    def __init__(
        self,
        n_inputs: int,
        n_nodes: int,
        libraries: List[Library],
        endpoint: Endpoint,
        fitness: Fitness,
        preprocessing: Preprocessing = None,
    ):
        super().__init__()
        if not isinstance(libraries, list):
            libraries = [libraries]
        self.model = KartezioCGP(
            n_inputs, n_nodes, libraries, endpoint, fitness, preprocessing
        )
        self.updatables = []

    def fit(
        self,
        n_iterations: int,
        x: List[Any],
        y: List[Any],
        callbacks: List[Callback] = [],
    ) -> "KartezioTrainer":
        # compile the Kartezio model
        self.model.initialize(n_iterations)

        # attach callbacks
        for callback in callbacks:
            callback.set_decoder(self.model.decoder)
            self.model.attach(callback)

        # preprocess data
        x = self.model.preprocess(x)

        # evolve the model
        return self.model.evolve(x, y)

    @property
    def decoder(self) -> DecoderCGP:
        return self.model.decoder

    def set_endpoint(self, endpoint: Endpoint):
        self.decoder.endpoint = endpoint

    def set_library(self, library: Library):
        self.decoder.library = library

    @property
    def strategy(self) -> OnePlusLambda:
        return self.model.evolver.strategy

    def set_n_children(self, n_children):
        self.strategy.n_children = n_children

    def set_required_fps(self, required_fps):
        self.strategy.set_required_fps(required_fps)

    def set_gamma(self, gamma):
        self.strategy.set_gamma(gamma)

    @property
    def mutation(self) -> MutationHandler:
        return self.strategy.mutation_handler

    def set_decay(self, decay: MutationDecay):
        self.mutation.set_decay(decay)

    def set_behavior(self, behavior: MutationBehavior):
        self.mutation.set_behavior(behavior)
        self.mutation.behavior.set_decoder(self.decoder)

    def set_mutation_rates(self, node_rate, out_rate):
        self.mutation.set_mutation_rates(node_rate, out_rate)

    def set_mutation_effect(self, effect):
        self.mutation.set_effect(effect)

    def set_mutation_edges(self, edges):
        self.mutation.set_edges(edges)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def print_python_class(self, class_name):
        python_writer = PythonClassWriter(self.decoder)
        python_writer.to_python_class(
            class_name, self.model.population.get_elite()
        )

    def display_elite(self):
        elite = self.model.elite
        print(elite[0])
        print(elite.outputs)

    def summary(self):
        # TODO: implement summary
        pass
