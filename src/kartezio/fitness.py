from abc import ABC

import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment

from kartezio.core.components.base import register
from kartezio.core.evolution import Fitness, KartezioFitness, KartezioMetric
from kartezio.metric import MetricMSE

# TODO: clear the fitness process


@jit(nopython=True)
def _label_overlap(x, y):
    """fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------
    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    Returns
    ------------
    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _intersection_over_union(masks_true, masks_pred):
    """intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    Returns
    ------------
    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


@register(Fitness, "average_precision")
class FitnessAP(Fitness):
    def __init__(self, reduction="mean", threshold=0.5, iou_factor=0.0):
        super().__init__(reduction)
        self.threshold = threshold
        self.iou_factor = iou_factor
        self.iou_fitness = FitnessIOU(reduction)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        ap = 1.0 - self.average_precision(y_true, y_pred)
        if self.iou_factor > 0.0:
            iou = self.iou_fitness.evaluate(y_true, y_pred) * self.iou_factor
            return ap + iou
        return ap

    def mask_ious(self, masks_true, masks_pred):
        """return best-matched masks"""
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
        n_min = min(iou.infos[0], iou.infos[1])
        costs = -(iou >= 0.5).astype(float) - iou / (2 * n_min)
        true_ind, pred_ind = linear_sum_assignment(costs)
        iout = np.zeros(masks_true.max())
        iout[true_ind] = iou[true_ind, pred_ind]
        preds = np.zeros(masks_true.max(), "int")
        preds[true_ind] = pred_ind + 1
        return iout, preds

    def average_precision(self, y_true, y_pred):
        """average precision estimation: AP = TP / (TP + FP + FN)
        This function is based heavily on the *fast* stardist matching functions
        (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)
        Parameters
        ------------

        masks_true: list of ND-arrays (int) or ND-array (int)
            where 0=NO masks; 1,2... are mask labels
        masks_pred: list of ND-arrays (int) or ND-array (int)
            ND-array (int) where 0=NO masks; 1,2... are mask labels
        Returns
        ------------
        ap: array [len(masks_true) x len(threshold)]
            average precision at thresholds
        tp: array [len(masks_true) x len(threshold)]
            number of true positives at thresholds
        fp: array [len(masks_true) x len(threshold)]
            number of false positives at thresholds
        fn: array [len(masks_true) x len(threshold)]
            number of false negatives at thresholds
        """

        n_images = len(y_true)
        ap = np.zeros(n_images, np.float32)
        tp = np.zeros(n_images, np.float32)
        fp = np.zeros(n_images, np.float32)
        fn = np.zeros(n_images, np.float32)
        n_true = np.array(list(map(np.max, y_true)))
        n_pred = np.array(list(map(np.max, y_pred)))
        for n in range(n_images):
            if n_pred[n]:
                iou = _intersection_over_union(y_true[n][0], y_pred[n][0])[1:, 1:]
                # tp[n, 0] = self._true_positive(iou, 0.5)
                n_min = min(iou.shape[0], iou.shape[1])
                costs = -(iou >= self.threshold).astype(float) - iou / (2 * n_min)
                true_ind, pred_ind = linear_sum_assignment(costs)
                match_ok = iou[true_ind, pred_ind] >= self.threshold
                tp[n] = match_ok.sum()
            fp[n] = n_pred[n] - tp[n]
            fn[n] = n_true[n] - tp[n]
            if tp[n] == 0:
                if n_true[n] == 0:
                    ap[n] = 1.0
                else:
                    ap[n] = 0.0
            else:
                ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])
        return ap


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
        n_images = len(y_true)
        ious = np.zeros(n_images, np.float32)
        for n in range(n_images):
            _y_true = y_true[n][0]
            _y_pred = y_pred[n][0]
            _y_pred[_y_pred > 0] = 1
            if np.sum(_y_true) == 0:
                _y_true = 1 - _y_true
                _y_pred = 1 - _y_pred
            intersection = np.logical_and(_y_true, _y_pred)
            union = np.logical_or(_y_true, _y_pred)
            ious[n] = np.sum(intersection) / np.sum(union)
        return 1.0 - ious

    def __init__(self, reduction="mean", multiprocessing=False):
        super().__init__(reduction, multiprocessing)


@register(Fitness, "intersection_over_union_2")
class FitnessIOU2(KartezioFitness):
    def __init__(self):
        super().__init__("IOU2", default_metric=registry.metrics.instantiate("IOU2"))


@register(Fitness, "mean_squared_error")
class FitnessMSE(Fitness):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.square(np.subtract(y_true, y_pred)).mean()

    def __init__(self, reduction="mean", multiprocessing=False):
        super().__init__(reduction, multiprocessing)


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
