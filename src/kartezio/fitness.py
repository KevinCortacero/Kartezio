from abc import ABC

import numpy as np

from kartezio.core.components.base import register
from kartezio.core.evolution import Fitness, KartezioFitness, KartezioMetric
from kartezio.metric import MetricMSE

# TODO: clear the fitness process


@register(Fitness, "mean_average_precision")
class FitnessAP(KartezioFitness):
    def __init__(self, thresholds=0.5):
        super().__init__(
            name=f"Average Precision ({thresholds})",
            symbol="AP",
            arity=1,
            default_metric=registry.metrics.instantiate("CAP", thresholds=thresholds),
        )


@register(Fitness, "count")
class FitnessCount(KartezioFitness):
    def __init__(self, secondary_metric: KartezioMetric = None):
        super().__init__(
            "Counting", default_metric=registry.metrics.instantiate("count")
        )
        if secondary_metric is not None:
            self.add_metric(secondary_metric)


@register(Fitness, "intersection_over_union")
class FitnessIOU(Fitness):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        _y_true = y_true[0]
        _y_pred = y_pred[0]
        _y_pred[_y_pred > 0] = 1
        if np.sum(_y_true) == 0:
            _y_true = 1 - _y_true
            _y_pred = 1 - _y_pred
        intersection = np.logical_and(_y_true, _y_pred)
        union = np.logical_or(_y_true, _y_pred)
        return 1 - np.sum(intersection) / np.sum(union)

    def __init__(self, reduction="mean", multiprocessing=False):
        super().__init__(reduction, multiprocessing)


@register(Fitness, "intersection_over_union_2")
class FitnessIOU2(KartezioFitness):
    def __init__(self):
        super().__init__("IOU2", default_metric=registry.metrics.instantiate("IOU2"))


@register(Fitness, "mean_squared_error")
class FitnessMSE(KartezioFitness):
    def __init__(self):
        super().__init__("Mean Squared Error", "MSE", 1, default_metric=MetricMSE())


@register(Fitness, "cross_entropy")
class FitnessCrossEntropy(KartezioFitness):
    def __init__(self, n_classes=2):
        super().__init__(
            "Cross-Entropy",
            "CE",
            n_classes,
            default_metric=registry.metrics.instantiate("cross_entropy"),
        )


@register(Fitness, "mcc")
class FitnessMCC(KartezioFitness):
    """
    author: Nghi Nguyen (2022)
    """

    def __init__(self):
        super().__init__("MCC", default_metric=registry.metrics.instantiate("MCC"))
