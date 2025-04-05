import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import disk

SHARPEN_KERNEL = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="int")
KERNEL_ROBERTS_X = np.array([[0, 1], [1, 0]], dtype=np.float32)
KERNEL_ROBERTS_Y = np.array([[-1, 0], [0, -1]], dtype=np.float32)
OPENCV_KERNEL_RANGE = [3, 5, 7, 9]
OPENCV_MIN_INTENSITY = 0
OPENCV_MAX_INTENSITY = 255
OPENCV_INTENSITY_RANGE = OPENCV_MAX_INTENSITY - OPENCV_MIN_INTENSITY
HITMISS_KERNEL = np.array(([0, 1, 0], [1, -1, 1], [0, 1, 0]), dtype="int")

KERNEL_EMBOSS = np.array(([-2, -1, 0], [-1, 1, 1], [0, 1, 2]), dtype="int")

KERNEL_KIRSCH_N = np.array(([5, 5, 5], [-3, 0, -3], [-3, -3, -3]), dtype="int")

KERNEL_KIRSCH_NE = np.array(
    ([-3, 5, 5], [-3, 0, 5], [-3, -3, -3]), dtype="int"
)

KERNEL_KIRSCH_E = np.array(([-3, -3, 5], [-3, 0, 5], [-3, -3, 5]), dtype="int")

KERNEL_KIRSCH_SE = np.array(
    ([-3, -3, -3], [-3, 0, 5], [-3, 5, 5]), dtype="int"
)

KERNEL_KIRSCH_S = np.array(([-3, -3, -3], [-3, 0, -3], [5, 5, 5]), dtype="int")

KERNEL_KIRSCH_SW = np.array(
    ([-3, -3, -3], [5, 0, -3], [5, 5, -3]), dtype="int"
)

KERNEL_KIRSCH_W = np.array(([5, -3, -3], [5, 0, -3], [5, -3, -3]), dtype="int")

KERNEL_KIRSCH_NW = np.array(
    ([5, 5, -3], [5, 0, -3], [-3, -3, -3]), dtype="int"
)

KERNEL_KIRSCH_COMPASS = [
    KERNEL_KIRSCH_N,
    KERNEL_KIRSCH_NE,
    KERNEL_KIRSCH_E,
    KERNEL_KIRSCH_SE,
    KERNEL_KIRSCH_S,
    KERNEL_KIRSCH_SW,
    KERNEL_KIRSCH_W,
    KERNEL_KIRSCH_NW,
]


def get_ksize_from_params(p):
    """
    Determines the kernel size based on the parameter value.

    Parameters:
    p (int): The parameter value to determine the kernel size.

    Returns:
    int: The kernel size from the predefined range.
    """
    idx = p // 64
    return OPENCV_KERNEL_RANGE[idx]


def disk_kernel(p):
    ksize = get_ksize_from_params(p)
    return disk(ksize // 2)


def get_hitmiss_kernel(ksize):
    ones = disk(ksize // 2, dtype=np.int8)
    if ksize == 3:
        ones[1, 1] = -1
    elif ksize == 5:
        ones[1:-1, 2] = -1
        ones[2, 1:-1] = -1
    else:
        negatives = disk((ksize - 2) // 2)
        ones[1:-1, 1:-1][negatives == 1] = -1
    return ones
