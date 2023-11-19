import cv2


def gradient_magnitude(gx, gy):
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel)
