import cv2
import czifile
import tifffile


def imread_rgb(filename):
    return cv2.cvtColor(imread_bgr(filename), cv2.COLOR_BGR2RGB)


def imread_bgr(filename):
    return cv2.imread(filename, cv2.IMREAD_COLOR)


def imread_gray(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def imread_tiff(filename):
    return tifffile.imread(filename)


def imread_czi(filename):
    return czifile.imread(filename)


def imwrite(filename, image):
    cv2.imwrite(filename, image)


def imwrite_tiff(filename, image, imagej=True, luts=None):
    if luts:
        tifffile.imwrite(
            filename,
            image,
            imagej=imagej,
            metadata={"mode": "composite"},
            ijmetadata=luts,
        )
