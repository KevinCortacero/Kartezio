from abc import ABC, abstractmethod

import numpy as np

from kartezio.model.components import BaseComponent


class Population(BaseComponent, ABC):
    def __init__(self, size):
        super().__init__("Population")
        self.size = size
        self.individuals = [None] * self.size
        self._fitness = {"fitness": np.zeros(self.size), "time": np.zeros(self.size)}

    def dumps(self) -> dict:
        return {}

    @abstractmethod
    def get_best_individual(self):
        pass

    def __getitem__(self, item):
        return self.individuals.__getitem__(item)

    def __setitem__(self, key, value):
        self.individuals.__setitem__(key, value)

    def set_time(self, individual, value):
        self._fitness["time"][individual] = value

    def set_fitness(self, fitness):
        self._fitness["fitness"] = fitness

    def has_best_fitness(self):
        return min(self.fitness) == 0.0

    @property
    def fitness(self):
        return self._fitness["fitness"]

    @property
    def time(self):
        return self._fitness["time"]

    @property
    def score(self):
        score_list = list(zip(self.fitness, self.time))
        return np.array(score_list, dtype=[("fitness", float), ("time", float)])
