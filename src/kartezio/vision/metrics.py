import numpy as np
from skimage.filters import sobel
from skimage.metrics import structural_similarity


def _intersection(a, b):
    return np.count_nonzero(np.logical_and(a, b))


def _union(a, b):
    return np.count_nonzero(np.logical_or(a, b))


def _tp(y_true, y_pred):
    return _intersection(y_true, y_pred)


def _fp(y_true, y_pred):
    return _intersection(np.logical_not(y_true), y_pred)


def _fn(y_true, y_pred):
    return _intersection(y_true, np.logical_not(y_pred))


def _tn(y_true, y_pred):
    return _intersection(np.logical_not(y_true), np.logical_not(y_pred))


def iou(y_true, y_pred):
    union = _union(y_true, y_pred)
    if union == 0:
        if np.count_nonzero(y_true) == 0:
            return 1
        else:
            return 0
    intersection = _intersection(y_true, y_pred)
    iou = intersection / union
    return iou


def dice(y_true, y_pred):
    total = np.count_nonzero(y_true) + np.count_nonzero(y_pred)
    if np.count_nonzero(y_true) == 0:
        if total == 0:
            return 1
    intersection = _intersection(y_true, y_pred)
    dice = 2.0 * intersection / total
    return dice


def precision(y_true, y_pred):
    true_positive = _tp(y_true, y_pred)
    false_positive = _fp(y_true, y_pred)
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )
    return precision


def recall(y_true, y_pred):
    true_positive = _tp(y_true, y_pred)
    false_negative = _fn(y_true, y_pred)
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )
    return recall


def f1(y_true, y_pred):
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    f1 = (
        2 * (precision_score * recall_score) / (precision_score + recall_score)
        if (precision_score + recall_score) > 0
        else 0
    )
    return f1


def accuracy(y_true, y_pred):
    true_positive = _tp(y_true, y_pred)
    false_positive = _fp(y_true, y_pred)
    false_negative = _fn(y_true, y_pred)
    true_negative = _tn(y_true, y_pred)
    accuracy = (
        (true_positive + true_negative)
        / (true_positive + false_positive + false_negative + true_negative)
        if (true_positive + false_positive + false_negative + true_negative)
        > 0
        else 0
    )
    return accuracy


def mse(y_true, y_pred):
    """
    L2 loss (Mean Squared Error) between y_true and y_pred.
    Penalizes large errors more than small ones, smoother output than MAE.
    """
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    """
    L1 loss (Mean Absolute Error) between y_true and y_pred.
    Penalizes all errors equally, more robust to outliers than MSE.
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE) between y_true and y_pred.
    Similar to MSE but in the same unit as the data.
    """
    return np.sqrt(mse(y_true, y_pred))


def ssim(y_true, y_pred):
    """
    Structural Similarity Index (SSIM) between y_true and y_pred.
    Measures the similarity between two images, considering luminance, contrast, and structure.
    """
    return structural_similarity(y_true, y_pred, multichannel=True)


def edge_mse(y_true, y_pred):
    """
    Edge Mean Squared Error (Edge MSE) between y_true and y_pred.
    Computes the MSE on the edges of the images.
    """
    y_true_edges = sobel(y_true)
    y_pred_edges = sobel(y_pred)
    return mse(y_true_edges, y_pred_edges)


def fft_mae(y_true, y_pred):
    """
    FFT Mean Absolute Error (FFT MAE) between y_true and y_pred.
    Computes the MAE in the frequency domain using FFT.
    """
    y_true_fft = np.fft.fft2(y_true)
    y_pred_fft = np.fft.fft2(y_pred)
    return mae(y_true_fft, y_pred_fft)


def psnr(y_true, y_pred):
    """
    Peak Signal-to-Noise Ratio (PSNR) between y_true and y_pred.
    Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
    """
    mse_value = mse(y_true, y_pred)
    if mse_value == 0:
        return float("inf")
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr_value


def balanced_metric(metric, y_true, y_pred, sensitivity=0.5, specificity=0.5):
    if sensitivity == 0 and specificity == 0:
        return metric(y_true, y_pred)
    if sensitivity == 0:
        return metric(y_true, y_pred) + specificity * precision(y_true, y_pred)
    if specificity == 0:
        return metric(y_true, y_pred) + sensitivity * recall(y_true, y_pred)
    return (
        metric(y_true, y_pred)
        + sensitivity * recall(y_true, y_pred)
        + specificity * precision(y_true, y_pred)
    )
