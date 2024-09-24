import numpy as np
from numba import jit


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
    dice = 2. * intersection / total
    return dice


def precision(y_true, y_pred):
    true_positive = _tp(y_true, y_pred)
    false_positive = _fp(y_true, y_pred)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    return precision


def recall(y_true, y_pred):
    true_positive = _tp(y_true, y_pred)
    false_negative = _fn(y_true, y_pred)
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    return recall


def f1(y_true, y_pred):
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    return f1


def accuracy(y_true, y_pred):
    true_positive = _tp(y_true, y_pred)
    false_positive = _fp(y_true, y_pred)
    false_negative = _fn(y_true, y_pred)
    true_negative = _tn(y_true, y_pred)
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative) if (true_positive + false_positive + false_negative + true_negative) > 0 else 0
    return accuracy


def iou_precision(y_true, y_pred):
    return iou(y_true, y_pred) + precision(y_true, y_pred)


def iou_recall(y_true, y_pred):
    return iou(y_true, y_pred) + recall(y_true, y_pred)


# Example usage
if __name__ == "__main__":
    # Example binary images (ground truth and prediction)
    y_true = np.random.randint(0, 2, (256, 256), dtype=np.uint8).ravel()
    y_pred = np.random.randint(0, 2, (256, 256), dtype=np.uint8).ravel()
    
    print("IoU:", iou(y_true, y_pred))
    print("Dice:", dice(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1:", f1(y_true, y_pred))
    print("Accuracy:", accuracy(y_true, y_pred))
    print("IoU + Precision:", iou_precision(y_true, y_pred))
    print("IoU + Recall:", iou_recall(y_true, y_pred))

    # time the functions
    import time
    start = time.time()
    print("IoU:", iou(y_true, y_pred))
    print("Time taken:", time.time() - start)
    print("Dice:", dice(y_true, y_pred))
    print("Precision:", precision(y_true, y_pred))
    print("Recall:", recall(y_true, y_pred))
    print("F1:", f1(y_true, y_pred))
    print("Accuracy:", accuracy(y_true, y_pred))
    print("IoU + Precision:", iou_precision(y_true, y_pred))
    print("IoU + Recall:", iou_recall(y_true, y_pred))
    print("Time taken:", time.time() - start)
    

