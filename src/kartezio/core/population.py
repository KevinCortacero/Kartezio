from abc import ABC, abstractmethod

import numpy as np

from kartezio.core.components.base import Component


class Population(Component, ABC):
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

    def dumps(self) -> dict:
        return {}

    def set_time(self, individual, value):
        self.score.time[individual] = value

    def set_fitness(self, fitness):
        self.score.fitness[1:] = fitness

    def set_raw_fitness(self, raw_fitness):
        if self.score.raw is None:
            self.score.raw = (
                np.ones((self.size, len(raw_fitness[0])), dtype=np.float32)
                * np.inf
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
        return np.array(
            score_list, dtype=[("fitness", float), ("time", float)]
        )
