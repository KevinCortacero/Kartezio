from abc import ABC
from typing import Dict

import numpy as np

from kartezio.core.components import KartezioComponent, fundamental, register


@fundamental()
class Population(KartezioComponent, ABC):
    class PopulationScore:
        def __init__(self, fitness, time):
            self.fitness = fitness
            self.time = time
            self.raw = None

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.individuals = [None] * self.size
        self.score = Population.PopulationScore(
            np.ones(self.size, dtype=np.float32) * np.inf,
            np.zeros(self.size, dtype=np.float32),
        )

    def set_time(self, individual, value):
        self.score.time[individual] = value

    def set_fitness(self, fitness):
        self.score.fitness[1:] = fitness

    def set_raw_fitness(self, raw_fitness):
        if self.score.raw is None:
            self.score.raw = (
                np.ones((self.size, len(raw_fitness[0])), dtype=np.float32) * np.inf
            )
        self.score.raw[1:] = raw_fitness

    def has_best_fitness(self):
        return min(self.get_fitness()) == 0.0

    def get_fitness(self):
        return self.score.fitness

    def get_time(self):
        return self.score.time

    def get_raw(self):
        return self.score.raw

    def get_score(self):
        score_list = list(zip(self.get_fitness(), self.get_time()))
        return np.array(score_list, dtype=[("fitness", float), ("time", float)])


class IndividualHistory:
    def __init__(self):
        self.genotype = None
        self.fitness = 0.0
        self.time = 0.0


class PopulationHistory:
    def __init__(self, n_individuals, changed: bool):
        self.individuals = {}
        for i in range(n_individuals):
            self.individuals[i] = IndividualHistory()
        self.changed = changed

    def get_best_fitness(self):
        return (
            self.individuals[0],
            self.individuals[0].fitness,
            self.individuals[0].time,
        )

    def get_individuals(self):
        return self.individuals.items()


@register(Population)
class PopulationWithElite(Population):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "PopulationWithElite":
        pass

    def __init__(self, n_children):
        super().__init__(1 + n_children)

    def get_elite(self):
        return self.individuals[0]

    def promote_new_parent(self):
        changed = False
        fitness = self.get_fitness()
        times = self.get_time()
        raw = self.get_raw()
        best_fitness_idx = np.argsort(self.get_score())[0]
        if best_fitness_idx != 0:
            changed = True
        self.individuals[0] = self.individuals[best_fitness_idx].clone()
        self.score.fitness[0] = fitness[best_fitness_idx]
        self.score.time[0] = times[best_fitness_idx]
        self.score.raw[0] = raw[best_fitness_idx]

        state = PopulationHistory(self.size, changed)

        for i in range(len(self.individuals)):
            state.individuals[i].genotype = self.individuals[i].clone()
            state.individuals[i].fitness = fitness[i]
            state.individuals[i].time = times[i]
        return state
