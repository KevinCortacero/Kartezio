from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from kartezio.core.components.base import Component


class Fitness(Component, ABC):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.mode = "train"

    def batch(self, y_true, y_pred):
        population_fitness = np.zeros(
            (len(y_pred), len(y_true)), dtype=np.float32
        )
        for idx_individual in range(len(y_pred)):
            population_fitness[idx_individual] = self.evaluate(
                y_true, y_pred[idx_individual]
            )
        return self._reduce(population_fitness)

    def _reduce(self, population_fitness):
        if self.reduction == "mean":
            return np.mean(population_fitness, axis=1)
        if self.reduction == "min":
            return np.min(population_fitness, axis=1)
        if self.reduction == "max":
            return np.max(population_fitness, axis=1)
        if self.reduction == "median":
            return np.median(population_fitness, axis=1)
        if self.reduction is None:
            return population_fitness

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Component":
        pass
