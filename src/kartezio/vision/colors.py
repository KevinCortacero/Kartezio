""" Color conversion functions """

import cv2
import numpy as np
from skimage import color


def rgb2hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def bgr2hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hsv2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def gray2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def rgb2hed(image):
    return (color.rgb2hed(image) * 255).astype(np.uint8)


def bgr2hed(image):
    return rgb2hed(bgr2rgb(image))
