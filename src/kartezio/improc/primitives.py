import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from numena.image.morphology import morph_fill
from numena.image.threshold import threshold_binary, threshold_tozero
from scipy.stats import kurtosis, skew
from skimage.morphology import remove_small_objects, remove_small_holes

from kartezio.improc.kernel import correct_ksize, SHARPEN_KERNEL, KERNEL_ROBERTS_X, KERNEL_ROBERTS_Y, gabor_kernel, \
    kernel_from_parameters
from kartezio.model.library import KLibrary
from kartezio.model.primitive import KPrimitive, KSignature
from kartezio.model.types import KType, TypeArray

@dataclass
class SignatureImage(KSignature):
    output_type: KType = field(init=False, default=TypeArray)


class LibraryDefaultOpenCV(KLibrary):
    def __init__(self):
        super().__init__(TypeArray)
    def create_primitive(self, name, symbol, arity, parameters, function):
        signature = SignatureImage(name, [TypeArray] * arity, parameters)
        primitive = KPrimitive(symbol, signature, function)
        self.add_primitive(primitive)


def f_max(x, args=None): return cv2.max(x[0], x[1])
def f_min(x, args=None): return cv2.min(x[0], x[1])
def f_mean(x, args=None): return cv2.addWeighted(x[0], 0.5, x[1], 0.5, 0)
def f_add(x, args=None): return cv2.add(x[0], x[1])
def f_sub(x, args=None): return cv2.subtract(x[0], x[1])
def f_bitwise_not(x, args=None): return cv2.bitwise_not(x[0])
def f_bitwise_or(x, args=None): return cv2.bitwise_or(x[0], x[1])
def f_bitwise_and(x, args=None): return cv2.bitwise_and(x[0], x[1])
def f_bitwise_and_mask(x, args=None): return cv2.bitwise_and(x[0], x[0], mask=x[1])
def f_bitwise_xor(x, args=None): return cv2.bitwise_xor(x[0], x[1])
def f_sqrt(x, args=None): return (cv2.sqrt((x[0] / 255.0).astype(np.float32)) * 255).astype(np.uint8)
# future: def f_sqrt(x, args=None): return cv2.convertScaleAbs(cv2.sqrt(image_test_f32))
def f_pow(x, args=None): return (cv2.pow((x[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)
# future: def f_pow(x, args=None): return cv2.convertScaleAbs(cv2.pow(x[0], 2))
def f_exp(x, args=None): return (cv2.exp((x[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)
# future: def f_pow(x, args=None): exp_image = cv2.exp(x[0], 2); exp_image[exp_image < 1] = 255; return exp_image
def f_log(x, args=None): return np.log1p(x[0]).astype(np.uint8)
# future: def f_log(x, args=None): return cv2.convertScaleAbs(np.log1p(image_test_f32))
def f_median_blur(x, args=None): return cv2.medianBlur(x[0], correct_ksize(args[0]))
def f_gaussian_blur(x, args=None):
    ksize = correct_ksize(args[0])
    return cv2.GaussianBlur(x[0], (ksize, ksize), 0)
def f_laplacian(x, args=None): return cv2.Laplacian(x[0], cv2.CV_64F).astype(np.uint8)
# future: def f_laplacian(x, args=None): return cv2.convertScaleAbs(cv2.Laplacian(x[0], cv2.CV_8U, ksize=1))
# future: def f_sobel_laplacian(x, args=None): return cv2.convertScaleAbs(cv2.Laplacian(x[0], cv2.CV_8U, ksize=3))

def f_sobel(x, args=None):
    ksize = correct_ksize(args[0])
    if args[1] < 128:
        return cv2.Sobel(x[0], cv2.CV_32F, 1, 0, ksize=ksize).astype(np.uint8)
    return cv2.Sobel(x[0], cv2.CV_32F, 0, 1, ksize=ksize).astype(np.uint8)

def f_sobel_new(x, args=None):
    dx, dy = cv2.spatialGradient(x[0])
    return cv2.convertScaleAbs(cv2.magnitude(dx.astype(np.float32), dy.astype(np.float32)))

def f_roberts(x, args=None):
    img = (x[0] / 255.0).astype(np.float32)
    h = cv2.filter2D(img, -1, KERNEL_ROBERTS_X)
    v = cv2.filter2D(img, -1, KERNEL_ROBERTS_Y)
    return (cv2.sqrt(cv2.pow(h, 2) + cv2.pow(v, 2)) * 255).astype(np.uint8)

def f_roberts_new(x, args=None):
    dx = cv2.filter2D(x[0], cv2.CV_32F, KERNEL_ROBERTS_X)
    dy = cv2.filter2D(x[0], cv2.CV_32F, KERNEL_ROBERTS_Y)
    return cv2.convertScaleAbs(cv2.magnitude(dx, dy))

def f_canny(x, args=None): return cv2.Canny(x[0], args[0], args[1])
def f_sharpen(x, args=None): return cv2.filter2D(x[0], -1, SHARPEN_KERNEL)
def f_gabor(x, args=None):
    gabor_k = gabor_kernel(11, args[0], args[1])
    return cv2.filter2D(x[0], -1, gabor_k)

def f_diff_gaussian(x, args=None):
    ksize = correct_ksize(args[0])
    return x[0] - cv2.GaussianBlur(x[0], (ksize, ksize), 0) + args[1]

def f_diff_gaussian_new(x, args=None):
    ksize = correct_ksize(args[0])
    image = np.array(x[0], dtype=np.int16)
    return image - cv2.GaussianBlur(image, (ksize, ksize), 0) + args[1]
    # return cv2.add(cv2.subtract(image, cv2.GaussianBlur(image, (ksize, ksize), 0)), args[1])

def f_absdiff(x, args=None): return 255 - cv2.absdiff(x[0], x[1])
def f_absdiff_new(x, args=None): return cv2.absdiff(x[0], x[1])

def f_fluo_tophat(x, args=None):
    kernel = kernel_from_parameters(args)
    img = cv2.morphologyEx(x[0], cv2.MORPH_TOPHAT, kernel, iterations=10)
    kur = np.mean(kurtosis(img, fisher=True))
    skew1 = np.mean(skew(img))
    if kur > 1 and skew1 > 1:
        p2, p98 = np.percentile(img, (15, 99.5), interpolation="linear")
    else:
        p2, p98 = np.percentile(img, (15, 100), interpolation="linear")
    output_img = np.clip(img, p2, p98)
    if p98 - p2 == 0:
        return (output_img * 255).astype(np.uint8)
    output_img = (output_img - p2) / (p98 - p2) * 255
    return output_img.astype(np.uint8)

def f_relative_diff(x, args=None):
    img = x[0]
    max_img = np.max(img)
    min_img = np.min(img)

    ksize = correct_ksize(args[0])
    gb = cv2.GaussianBlur(img, (ksize, ksize), 0)
    gb = np.float32(gb)

    img = np.divide(img, gb + 1e-15, dtype=np.float32)
    img = cv2.normalize(img, img, max_img, min_img, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

def f_erode(x, args=None): return cv2.erode(x[0], kernel_from_parameters(args))
def f_dilate(x, args=None): return cv2.dilate(x[0], kernel_from_parameters(args))
def f_open(x, args=None): return cv2.morphologyEx(x[0], cv2.MORPH_OPEN, kernel_from_parameters(args))
def f_close(x, args=None): return cv2.morphologyEx(x[0], cv2.MORPH_CLOSE, kernel_from_parameters(args))
def f_morph_gradient(x, args=None): return cv2.morphologyEx(x[0], cv2.MORPH_GRADIENT, kernel_from_parameters(args))
def f_morph_tophat(x, args=None): return cv2.morphologyEx(x[0], cv2.MORPH_TOPHAT, kernel_from_parameters(args))
def f_morph_blackhat(x, args=None): return cv2.morphologyEx(x[0], cv2.MORPH_BLACKHAT, kernel_from_parameters(args))
def f_fill(x, args=None): return morph_fill(x[0])
def f_remove_small_objects(x, args=None): return remove_small_objects(x[0] > 0, args[0]).astype(np.uint8)
def f_remove_small_holes(x, args=None): return remove_small_holes(x[0] > 0, args[0]).astype(np.uint8)
def f_threshold(x, args=None):
    if args[0] < 128:
        return threshold_binary(x[0], args[1])
    return threshold_tozero(x[0], args[1])

def f_threshold_at_1(x, args=None):
    if args[0] < 128:
        return threshold_binary(x[0], 1)
    return threshold_tozero(x[0], 1)

def f_distance_transform(x, args=None):
    return cv2.normalize(
        cv2.distanceTransform(x[0].copy(), cv2.DIST_L2, 3),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U,
        )

def f_distance_transform_thresh(x, args=None):
    d = cv2.normalize(
        cv2.distanceTransform(x[0].copy(), cv2.DIST_L2, 3),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U,
    )
    return threshold_binary(d, args[0])


def f_bin_inrange(x, args=None):
    lower = int(min(args[0], args[1]))
    upper = int(max(args[0], args[1]))
    return cv2.inRange(x[0], lower, upper)

def f_inrange(x, args=None):
    lower = int(min(args[0], args[1]))
    upper = int(max(args[0], args[1]))
    return cv2.bitwise_and(
        x[0],
        x[0],
        mask=cv2.inRange(x[0], lower, upper),
    )

library_opencv = LibraryDefaultOpenCV()
library_opencv.create_primitive("Max", "max", 2, 0, f_max)
library_opencv.create_primitive("Min", "min", 2, 0, f_min)
library_opencv.create_primitive("Mean", "mean", 2, 0, f_mean)
library_opencv.create_primitive("Add", "add", 2, 0, f_add)
library_opencv.create_primitive("Subtract", "sub", 2, 0, f_sub)
library_opencv.create_primitive("Bitwise Not", "not", 1, 0, f_bitwise_not)
library_opencv.create_primitive("Bitwise Or", "or", 2, 0, f_bitwise_or)
library_opencv.create_primitive("Bitwise And", "and", 2, 0, f_bitwise_and)
library_opencv.create_primitive("Bitwise And Mask", "andm", 2, 0, f_bitwise_and_mask)
library_opencv.create_primitive("Bitwise Xor", "xor", 2, 0, f_bitwise_xor)
library_opencv.create_primitive("Square Root", "sqrt", 1, 0, f_sqrt)
library_opencv.create_primitive("Power 2", "pow", 1, 0, f_pow)
library_opencv.create_primitive("Exp", "exp", 1, 0, f_exp)
library_opencv.create_primitive("Log", "log", 1, 0, f_log)
library_opencv.create_primitive("Median Blur", "blrm", 1, 1, f_median_blur)
library_opencv.create_primitive("Gaussian Blur", "blrg", 1, 1, f_gaussian_blur)
library_opencv.create_primitive("Laplacian", "lplc", 1, 0, f_laplacian)
library_opencv.create_primitive("Sobel", "sobl", 1, 2, f_sobel)
library_opencv.create_primitive("Roberts", "rbrt", 1, 1, f_roberts_new)
library_opencv.create_primitive("Canny", "cany", 1, 2, f_canny)
library_opencv.create_primitive("Sharpen", "shrp", 1, 0, f_sharpen)
library_opencv.create_primitive("Gabor", "gabr", 1, 2, f_gabor)
library_opencv.create_primitive("Subtract Gaussian", "absd", 1, 2, f_diff_gaussian)
library_opencv.create_primitive("Absolute Difference", "abs2", 2, 0, f_absdiff)
library_opencv.create_primitive("Fluo TopHat", "fluo", 1, 2, f_fluo_tophat)
library_opencv.create_primitive("Relative Difference", "reld", 1, 1, f_relative_diff)
library_opencv.create_primitive("Erode", "erod", 1, 2, f_erode)
library_opencv.create_primitive("Dilate", "dilt", 1, 2, f_dilate)
library_opencv.create_primitive("Open", "open", 1, 2, f_open)
library_opencv.create_primitive("Close", "clse", 1, 2, f_close)
library_opencv.create_primitive("Morph Gradient", "mgrd", 1, 2, f_morph_gradient)
library_opencv.create_primitive("Morph Tophat", "mtht", 1, 2, f_morph_tophat)
library_opencv.create_primitive("Morph BlackHat", "mbht", 1, 2, f_morph_blackhat)
library_opencv.create_primitive("Morph Fill", "fill", 1, 0, f_fill)
library_opencv.create_primitive("Remove Small Objects", "rmso", 1, 1, f_remove_small_objects)
library_opencv.create_primitive("Remove Small Holes", "rmsh", 1, 1, f_remove_small_holes)
library_opencv.create_primitive("Threshold", "trh", 1, 2, f_threshold)
library_opencv.create_primitive("Threshold at 1", "trh1", 1, 1, f_threshold_at_1)
library_opencv.create_primitive("Distance Transform", "dtrf", 1, 1, f_distance_transform)
library_opencv.create_primitive("Distance Transform Threshold", "dttr", 1, 2, f_distance_transform_thresh)
library_opencv.create_primitive("Binary In Range", "brng", 1, 2, f_bin_inrange)
library_opencv.create_primitive("In Range", "rng", 1, 2, f_inrange)
