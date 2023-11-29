from abc import ABC
from typing import List

import numpy as np

from kartezio.model.components import BaseNode
from kartezio.model.types import Score, ScoreList


class KartezioMetric(BaseNode, ABC):
    def __init__(
        self,
        name: str,
        symbol: str,
        arity: int,
    ):
        super().__init__(name, symbol, arity, 0)

    def _to_json_kwargs(self) -> dict:
        pass


class KMetric(BaseNode, ABC):
    pass


MetricList = List[KartezioMetric]


class Fitness(BaseNode, ABC):
    def __init__(self, fn, reduction="mean", multiprocessing=False):
        super().__init__(fn)
        self.reduction = reduction
        self.multiprocessing = multiprocessing

    def batch(self, y_true, y_pred):
        population_fitness = np.zeros((len(y_pred), len(y_true)), dtype=np.float32)
        if self.multiprocessing:
            pass
        else:
            for idx_individual in range(len(y_pred)):
                for idx_image in range(len(y_true)):
                    _y_true = y_true[idx_image].copy()
                    _y_pred = y_pred[idx_individual][idx_image]
                    population_fitness[idx_individual, idx_image] = self.evaluate(
                        _y_true, _y_pred
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

    def evaluate(self, y_true, y_pred):
        return self._fn(y_true, y_pred)


class KartezioFitness(BaseNode, ABC):
    def __init__(
        self,
        name: str,
        symbol: str,
        arity: int,
        default_metric: KartezioMetric = None,
    ):
        super().__init__(name, arity, 0)
        self.metrics: MetricList = []
        if default_metric:
            self.add_metric(default_metric)

    def add_metric(self, metric: KartezioMetric):
        self.metrics.append(metric)

    def call(self, y_true, y_pred) -> ScoreList:
        scores: ScoreList = []
        for yi_pred in y_pred:
            scores.append(self.compute_one(y_true, yi_pred))
        return scores

    def compute_one(self, y_true, y_pred) -> Score:
        score = 0.0
        y_size = len(y_true)
        for i in range(y_size):
            _y_true = y_true[i].copy()
            _y_pred = y_pred[i]
            score += self.__fitness_sum(_y_true, _y_pred)
        return Score(score / y_size)

    def __fitness_sum(self, y_true, y_pred) -> Score:
        score = Score(0.0)
        for metric in self.metrics:
            score += metric.call(y_true, y_pred)
        return score

    def _to_json_kwargs(self) -> dict:
        pass
