import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment

from kartezio.core.components.base import register
from kartezio.core.evolution import KartezioMetric, KMetric
from kartezio.core.types import Score


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


@register(KMetric, "mean_average_precision")
class MetricCellpose(KMetric):
    """
    from MouseLand/cellpose:
    https://github.com/MouseLand/cellpose/blob/5cc03de9c2aa342d4b4469ff476ca04541b63414/cellpose/metrics.py
    """

    def __init__(self, thresholds=0.5):
        super().__init__("Cellpose Average Precision")
        self.thresholds = thresholds
        if not isinstance(self.thresholds, list) and not isinstance(
            self.thresholds, np.ndarray
        ):
            self.thresholds = [self.thresholds]
        self.n_thresholds = len(self.thresholds)

    def call(self, y_true: np.ndarray, y_pred: np.ndarray) -> Score:
        _y_true = y_true[0]
        _y_pred = y_pred["labels"]
        ap, tp, fp, fn = self.average_precision(_y_true, _y_pred)
        return 1.0 - ap[0]

    def aggregated_jaccard_index(self, masks_true, masks_pred):
        """AJI = intersection of all matched masks / union of all masks

        Parameters
        ------------

        masks_true: list of ND-arrays (int) or ND-array (int)
            where 0=NO masks; 1,2... are mask labels
        masks_pred: list of ND-arrays (int) or ND-array (int)
            ND-array (int) where 0=NO masks; 1,2... are mask labels
        Returns
        ------------
        aji : aggregated jaccard index for each set of masks
        """

        aji = np.zeros(len(masks_true))
        for n in range(len(masks_true)):
            iout, preds = self.mask_ious(masks_true[n], masks_pred[n])
            inds = np.arange(0, masks_true[n].max(), 1, int)
            overlap = self._label_overlap(masks_true[n], masks_pred[n])
            union = np.logical_or(masks_true[n] > 0, masks_pred[n] > 0).sum()
            overlap = overlap[inds[preds > 0] + 1, preds[preds > 0].astype(int)]
            aji[n] = overlap.sum() / union
        return aji

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

    def average_precision(self, masks_true, masks_pred):
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
        not_list = False
        if not isinstance(masks_true, list):
            masks_true = [masks_true]
            masks_pred = [masks_pred]
            not_list = True

        ap = np.zeros((len(masks_true), self.n_thresholds), np.float32)
        tp = np.zeros((len(masks_true), self.n_thresholds), np.float32)
        fp = np.zeros((len(masks_true), self.n_thresholds), np.float32)
        fn = np.zeros((len(masks_true), self.n_thresholds), np.float32)
        n_true = np.array(list(map(np.max, masks_true)))
        n_pred = np.array(list(map(np.max, masks_pred)))
        for n in range(len(masks_true)):
            #  _,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
            if n_pred[n] > 0:
                iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
                for k, th in enumerate(self.thresholds):
                    tp[n, k] = self._true_positive(iou, th)
            fp[n] = n_pred[n] - tp[n]
            fn[n] = n_true[n] - tp[n]
            if tp[n] == 0:
                if n_true[n] == 0:
                    ap[n] = 1.0
                else:
                    ap[n] = 0.0
            else:
                ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

        if not_list:
            ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
        return ap, tp, fp, fn

    def _true_positive(self, iou, th):
        """true positive at threshold th

        Parameters
        ------------
        iou: float, ND-array
            array of IOU pairs
        th: float
            threshold on IOU for positive label
        Returns
        ------------
        tp: float
            number of true positives at threshold
        """
        n_min = min(iou.infos[0], iou.infos[1])
        costs = -(iou >= th).astype(float) - iou / (2 * n_min)
        true_ind, pred_ind = linear_sum_assignment(costs)
        match_ok = iou[true_ind, pred_ind] >= th
        tp = match_ok.sum()
        return tp


@register(KMetric, "intersection_over_union")
class MetricIOU(KartezioMetric):
    def __init__(self):
        super().__init__("Intersection Over Union", symbol="IOU", arity=1)

    def call(self, y_true: np.ndarray, y_pred: np.ndarray) -> Score:
        _y_true = y_true[0]
        _y_pred = y_pred["mask"]
        _y_pred[_y_pred > 0] = 1
        if np.sum(_y_true) == 0:
            _y_true = 1 - _y_true
            _y_pred = 1 - _y_pred
        intersection = np.logical_and(_y_true, _y_pred)
        union = np.logical_or(_y_true, _y_pred)
        score = Score(1.0 - np.sum(intersection) / np.sum(union))
        return score


@register(KMetric, "intersection_over_union_2")
class MetricIOU2(KartezioMetric):
    def call(self, y_true: np.ndarray, y_pred: np.ndarray) -> Score:
        _y_true = y_true[0]
        _y_pred = y_pred["mask"]
        _y_pred[_y_pred > 0] = 1
        _y_true[_y_true > 0] = 1
        iou = _intersection_over_union(_y_true, _y_pred)
        len_true, len_pred = iou.infos
        if len_pred == 1 and len_true > 1:
            return Score(1.0)
        return Score(1.0 - iou[-1, -1])


@register(KMetric, "mean_squared_error")
class MetricMSE(KartezioMetric):
    def __init__(self):
        super().__init__("Mean Squared Error", symbol="MSE", arity=1)

    def call(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.square(np.subtract(y_true, y_pred)).mean()


@register(KMetric, "precision")
class MetricPrecision(KartezioMetric):
    def call(self, y_true: np.ndarray, y_pred: np.ndarray):
        _y_pred = y_pred["mask"]
        _y_pred[_y_pred > 0] = 1
        pred_1 = _y_pred == 1
        pred_0 = _y_pred == 0
        label_1 = y_true == 1
        label_0 = y_true == 0

        TP = (pred_1 & label_1).sum()
        FP = (pred_1 & label_0).sum()
        FN = (pred_0 & label_0).sum()

        if TP == 0 and FP == 0 and FN == 0:
            precision = 1.0
        elif TP == 0 and (FP > 0 or FN > 0):
            precision = 0.0
        else:
            precision = TP / (TP + FP)

        return precision
