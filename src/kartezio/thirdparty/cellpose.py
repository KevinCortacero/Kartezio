import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment


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


def mask_ious(masks_true, masks_pred):
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


def cellpose_ap(y_true, y_pred, threshold):
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
    n_true = np.array([len(np.unique(mask)) - 1 for mask in y_true])
    n_pred = np.array([len(np.unique(mask)) - 1 for mask in y_pred])
    for n in range(n_images):
        if n_pred[n]:
            iou = _intersection_over_union(y_true[n][0], y_pred[n][0])[1:, 1:]
            n_min = min(iou.shape[0], iou.shape[1])
            costs = -(iou >= threshold).astype(float) - iou / (2 * n_min)
            true_ind, pred_ind = linear_sum_assignment(costs)
            match_ok = iou[true_ind, pred_ind] >= threshold
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
