from kartezio.metric import MetricMSE

from kartezio.model.evolution import KartezioFitness, KartezioMetric
from kartezio.model.registry import registry

# TODO: clear the fitness process


def register_fitness():
    """Force decorators to wrap KartezioFitness"""
    print(f"[Kartezio - INFO] -  {len(registry.fitness.list())} fitness registered.")


@registry.fitness.add("AP")
class FitnessAP(KartezioFitness):
    def __init__(self, thresholds=0.5):
        super().__init__(
            name=f"Average Precision ({thresholds})",
            symbol="AP",
            arity=1,
            default_metric=registry.metrics.instantiate("CAP", thresholds=thresholds),
        )


@registry.fitness.add("count")
class FitnessCount(KartezioFitness):
    def __init__(self, secondary_metric: KartezioMetric = None):
        super().__init__(
            "Counting", default_metric=registry.metrics.instantiate("count")
        )
        if secondary_metric is not None:
            self.add_metric(secondary_metric)


@registry.fitness.add("IOU")
class FitnessIOU(KartezioFitness):
    def __init__(self):
        super().__init__(
            "Intersection Over Union",
            "IOU",
            1,
            default_metric=registry.metrics.instantiate("IOU"),
        )


@registry.fitness.add("IOU2")
class FitnessIOU2(KartezioFitness):
    def __init__(self):
        super().__init__("IOU2", default_metric=registry.metrics.instantiate("IOU2"))


@registry.fitness.add("MSE")
class FitnessMSE(KartezioFitness):
    def __init__(self):
        super().__init__("Mean Squared Error", "MSE", 1, default_metric=MetricMSE())


@registry.fitness.add("CE")
class FitnessCrossEntropy(KartezioFitness):
    def __init__(self, n_classes=2):
        super().__init__(
            "Cross-Entropy",
            "CE",
            n_classes,
            default_metric=registry.metrics.instantiate("cross_entropy"),
        )


@registry.fitness.add("MCC")
class FitnessMCC(KartezioFitness):
    """
    author: Nghi Nguyen (2022)
    """

    def __init__(self):
        super().__init__("MCC", default_metric=registry.metrics.instantiate("MCC"))
