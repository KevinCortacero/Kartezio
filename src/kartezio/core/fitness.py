from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

from kartezio.core.components import register
from kartezio.evolution.fitness import Fitness
from kartezio.thirdparty.cellpose import cellpose_ap
from kartezio.vision.metrics import balanced_metric, iou


@register(Fitness, "average_precision")
class FitnessAP(Fitness):
    def __init__(self, reduction="mean", threshold=0.5, iou_factor=0.0):
        super().__init__(reduction)
        self.threshold = threshold
        self.iou_factor = float(iou_factor)
        self.iou_fitness = FitnessIOU(reduction)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        ap = 1.0 - cellpose_ap(y_true, y_pred, self.threshold)
        if self.iou_factor > 0.0:
            iou = self.iou_fitness.evaluate(y_true, y_pred) * self.iou_factor
            return ap + iou
        return ap

    def __to_dict__(self) -> Dict:
        return {
            "name": "average_precision",
            "args": {
                "reduction": self.reduction,
                "threshold": self.threshold,
                "iou_factor": self.iou_factor,
            },
        }


@register(Fitness, "intersection_over_union")
class FitnessIOU(Fitness):
    def __init__(self, reduction="mean", balance=None):
        super().__init__(reduction)
        self.balance = balance

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        n_images = len(y_true)
        ious = np.zeros(n_images, np.float32)
        for n in range(n_images):
            _y_true = y_true[n][0].ravel()
            _y_pred = y_pred[n][0].ravel()
            _y_pred[_y_pred > 0] = 1
            if self.balance is None:
                ious[n] = 1.0 - iou(_y_true, _y_pred)
            elif self.balance == "sensitivity":
                ious[n] = 2.0 - balanced_metric(
                    iou, _y_true, _y_pred, sensitivity=1.0, specificity=0.0
                )
            elif self.balance == "specificity":
                ious[n] = 2.0 - balanced_metric(
                    iou, _y_true, _y_pred, sensitivity=0.0, specificity=1.0
                )
            elif self.balance == "balanced":
                ious[n] = 2.0 - balanced_metric(
                    iou, _y_true, _y_pred, sensitivity=0.5, specificity=0.5
                )
        return ious

    def __to_dict__(self) -> Dict:
        return {
            "name": "intersection_over_union",
            "args": {"reduction": self.reduction, "balance": self.balance},
        }


@register(Fitness, "mean_squared_error")
class FitnessMSE(Fitness):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        n_images = len(y_true)
        mse_values = np.zeros(n_images, np.float32)

        for n in range(n_images):
            _y_true = y_true[n][0]
            _y_pred = y_pred[n][0]

            # Compute Mean Squared Error
            mse_values[n] = np.mean((_y_true - _y_pred) ** 2)

        return mse_values

    def __init__(self, reduction="mean", multiprocessing=False):
        super().__init__(reduction, multiprocessing)
