import cv2
import numpy as np

SHARPEN_KERNEL = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="int")
KERNEL_ROBERTS_X = np.array(([0, 1], [-1, 0]), dtype="int")
KERNEL_ROBERTS_Y = np.array(([1, 0], [0, -1]), dtype="int")
OPENCV_MIN_KERNEL_SIZE = 3
OPENCV_MAX_KERNEL_SIZE = 11
OPENCV_KERNEL_RANGE = OPENCV_MAX_KERNEL_SIZE - OPENCV_MIN_KERNEL_SIZE
OPENCV_MIN_INTENSITY = 0
OPENCV_MAX_INTENSITY = 255
OPENCV_INTENSITY_RANGE = OPENCV_MAX_INTENSITY - OPENCV_MIN_INTENSITY
HITMISS_KERNEL = np.array(([0, 1, 0], [1, -1, 1], [0, 1, 0]), dtype="int")

KERNEL_SCALE = OPENCV_KERNEL_RANGE / OPENCV_INTENSITY_RANGE

GABOR_SIGMAS = [
    0.1,
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
]
GABOR_THETAS = np.arange(0, 2, step=1.0 / 8) * np.pi
GABOR_LAMBDS = [
    0.1,
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
]
GABOR_GAMMAS = np.arange(0.0625, 1.001, step=1.0 / 16)


KERNEL_EMBOSS = np.array(([-2, -1, 0], [-1, 1, 1], [0, 1, 2]), dtype="int")

KERNEL_KIRSCH_N = np.array(([5, 5, 5], [-3, 0, -3], [-3, -3, -3]), dtype="int")

KERNEL_KIRSCH_NE = np.array(([-3, 5, 5], [-3, 0, 5], [-3, -3, -3]), dtype="int")

KERNEL_KIRSCH_E = np.array(([-3, -3, 5], [-3, 0, 5], [-3, -3, 5]), dtype="int")

KERNEL_KIRSCH_SE = np.array(([-3, -3, -3], [-3, 0, 5], [-3, 5, 5]), dtype="int")

KERNEL_KIRSCH_S = np.array(([-3, -3, -3], [-3, 0, -3], [5, 5, 5]), dtype="int")

KERNEL_KIRSCH_SW = np.array(([-3, -3, -3], [5, 0, -3], [5, 5, -3]), dtype="int")

KERNEL_KIRSCH_W = np.array(([5, -3, -3], [5, 0, -3], [5, -3, -3]), dtype="int")

KERNEL_KIRSCH_NW = np.array(([5, 5, -3], [5, 0, -3], [-3, -3, -3]), dtype="int")

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


def clamp_ksize(ksize):
    if ksize < OPENCV_MIN_KERNEL_SIZE:
        return OPENCV_MIN_KERNEL_SIZE
    if ksize > OPENCV_MAX_KERNEL_SIZE:
        return OPENCV_MAX_KERNEL_SIZE
    return ksize


def remap_ksize(ksize):
    return int(round(ksize * KERNEL_SCALE + OPENCV_MIN_KERNEL_SIZE))


def unodd_ksize(ksize):
    if ksize % 2 == 0:
        return ksize + 1
    return ksize


def correct_ksize(ksize):
    ksize = remap_ksize(ksize)
    ksize = clamp_ksize(ksize)
    ksize = unodd_ksize(ksize)
    return ksize


def ellipse_kernel(ksize):
    ksize = correct_ksize(ksize)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


def cross_kernel(ksize):
    ksize = correct_ksize(ksize)
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))


def rect_kernel(ksize):
    ksize = correct_ksize(ksize)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))


def gabor_kernel(ksize, p1, p2):
    ksize = clamp_ksize(ksize)
    ksize = unodd_ksize(ksize)
    p1_bin = "{0:08b}".format(p1)
    p2_bin = "{0:08b}".format(p2)

    sigma = GABOR_SIGMAS[int(p1_bin[:4], 2)]
    theta = GABOR_THETAS[int(p1_bin[4:], 2)]
    lambd = GABOR_LAMBDS[int(p2_bin[:4], 2)]
    gamma = GABOR_GAMMAS[int(p2_bin[4:], 2)]

    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)


def kernel_from_parameters(p):
    # 50%
    if p[1] < 128:
        return ellipse_kernel(p[0])
    # 25%
    if p[1] < 192:
        return ellipse_kernel(p[0])
    # 25%
    return ellipse_kernel(p[0])
