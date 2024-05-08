from abc import ABC, abstractmethod

import numpy as np

from kartezio.core.components.base import Component


class Population(Component, ABC):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.individuals = [None] * self.size
        self._fitness = {
            "fitness": np.ones(self.size, dtype=np.float32) * np.inf,
            "time": np.zeros(self.size, dtype=np.float32),
        }

    def dumps(self) -> dict:
        return {}

    def set_time(self, individual, value):
        self._fitness["time"][individual] = value

    def set_fitness(self, fitness):
        self._fitness["fitness"][1:] = fitness

    def has_best_fitness(self):
        return min(self.get_fitness()) == 0.0

    def get_fitness(self):
        return self._fitness["fitness"]

    def get_time(self):
        return self._fitness["time"]

    def get_score(self):
        score_list = list(zip(self.get_fitness(), self.get_time()))
        return np.array(
            score_list, dtype=[("fitness", float), ("time", float)]
        )
