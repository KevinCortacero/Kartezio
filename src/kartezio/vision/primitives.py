from typing import List

import cv2
import numpy as np
import toml
from numena.image.morphology import morph_fill
from numena.image.threshold import threshold_binary, threshold_tozero
from scipy.stats import kurtosis, skew
from skimage.morphology import remove_small_holes, remove_small_objects

from kartezio.core.components.base import Components, register
from kartezio.core.components.decoder import SequentialDecoder
from kartezio.core.components.library import Library
from kartezio.core.components.primitive import Primitive
from kartezio.core.types import TypeArray
from kartezio.vision.common import convolution, gradient_magnitude
from kartezio.vision.kernel import (
    HITMISS_KERNEL,
    KERNEL_EMBOSS,
    KERNEL_KIRSCH_COMPASS,
    KERNEL_ROBERTS_X,
    KERNEL_ROBERTS_Y,
    SHARPEN_KERNEL,
    correct_ksize,
    gabor_kernel,
    kernel_from_parameters,
)


@register(Library, "opencv_library")
class LibraryDefaultOpenCV(Library):
    def __init__(self):
        super().__init__(TypeArray)


@register(Primitive, "max")
class Max(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.max(x[0], x[1])


@register(Primitive, "min")
class Min(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.min(x[0], x[1])


@register(Primitive, "mean")
class Mean(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.addWeighted(x[0], 0.5, x[1], 0.5, 0)


@register(Primitive, "add")
class Add(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.add(x[0], x[1])


@register(Primitive, "subtract")
class Subtract(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.subtract(x[0], x[1])


@register(Primitive, "bitwise_not")
class BitwiseNot(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_not(x[0])


@register(Primitive, "bitwise_or")
class BitwiseOr(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_or(x[0], x[1])


@register(Primitive, "bitwise_and")
class BitwiseAnd(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_and(x[0], x[1])


@register(Primitive, "bitwise_and_mask")
class BitwiseAndMask(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_and(x[0], x[0], mask=x[1])


@register(Primitive, "bitwise_xor")
class BitwiseXor(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_xor(x[0], x[1])


@register(Primitive, "sqrt")
class Sqrt(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.convertScaleAbs(cv2.sqrt(x[0].astype(np.float32)))


@register(Primitive, "pow")
class Pow(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.convertScaleAbs(cv2.pow(x[0], 2))


@register(Primitive, "exp")
class Exp(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        # future:
        # exp_image = cv2.exp(x[0])
        # exp_image[exp_image < 1] = 255
        # return exp_image
        return (cv2.exp((x[0] / 255.0).astype(np.float32)) * 255).astype(
            np.uint8
        )


@register(Primitive, "log")
class Log(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.convertScaleAbs(np.log1p(x[0].astype(np.float32)))


@register(Primitive, "median_blur")
class MedianBlur(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.medianBlur(x[0], correct_ksize(args[0]))


@register(Primitive, "gaussian_blur")
class GaussianBlur(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        ksize = correct_ksize(args[0])
        return cv2.GaussianBlur(x[0], (ksize, ksize), 0)


@register(Primitive, "laplacian")
class Laplacian(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.convertScaleAbs(cv2.Laplacian(x[0], cv2.CV_8U, ksize=1))


# future: def f_sobel_laplacian(x, args=None): return cv2.convertScaleAbs(cv2.Laplacian(x[0], cv2.CV_8U, ksize=3))


@register(Primitive, "sobel")
class Sobel(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        gx, gy = cv2.spatialGradient(x[0])
        return gradient_magnitude(gx.astype(np.float32), gy.astype(np.float32))


@register(Primitive, "roberts")
class Roberts(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        gx = convolution(x[0], KERNEL_ROBERTS_X)
        gy = convolution(x[0], KERNEL_ROBERTS_Y)
        return gradient_magnitude(gx.astype(np.float32), gy.astype(np.float32))


@register(Primitive, "canny")
class Canny(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        # return cv2.Canny(x[0], args[0], args[1])
        return cv2.Canny(x[0], args[0], args[0] * 3)


@register(Primitive, "sharpen")
class Sharpen(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return convolution(x[0], SHARPEN_KERNEL)


@register(Primitive, "gaussian_diff")
class GaussianDiff(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        ksize = correct_ksize(args[0])
        image = np.array(x[0], dtype=np.int16)
        return cv2.convertScaleAbs(
            image - cv2.GaussianBlur(image, (ksize, ksize), 0) + args[1]
        )


@register(Primitive, "abs_diff")
class AbsDiff(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.absdiff(x[0], x[1])


@register(Primitive, "erode")
class Erode(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.erode(x[0], kernel_from_parameters(args))


@register(Primitive, "dilate")
class Dilate(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.dilate(x[0], kernel_from_parameters(args))


@register(Primitive, "open")
class Open(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_OPEN, kernel_from_parameters(args)
        )


@register(Primitive, "close")
class Close(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_CLOSE, kernel_from_parameters(args)
        )


@register(Primitive, "gradient")
class Gradient(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_GRADIENT, kernel_from_parameters(args)
        )


def f_morph_gradient(x, args=None):
    return cv2.morphologyEx(
        x[0], cv2.MORPH_GRADIENT, kernel_from_parameters(args)
    )


@register(Primitive, "top_hat")
class TopHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_TOPHAT, kernel_from_parameters(args)
        )


@register(Primitive, "black_hat")
class BlackHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_BLACKHAT, kernel_from_parameters(args)
        )


@register(Primitive, "hit_miss")
class HitMiss(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_HITMISS, HITMISS_KERNEL)


@register(Primitive, "fill")
class Fill(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return morph_fill(x[0])


@register(Primitive, "rm_small_objects")
class RmSmallObjects(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return remove_small_objects(x[0] > 0, args[0]).astype(np.uint8)


@register(Primitive, "rm_small_holes")
class RmSmallHoles(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return remove_small_holes(x[0] > 0, args[0]).astype(np.uint8)


@register(Primitive, "binary_threshold")
class BinaryThreshold(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return threshold_binary(x[0], args[0])


@register(Primitive, "to_zero_threshold")
class ToZeroThreshold(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return threshold_tozero(x[0], args[0])


@register(Primitive, "binarize")
class Binarize(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return threshold_binary(x[0], 1)


@register(Primitive, "fluo_tophat")
class FluoTopHat(Primitive):
    """from https://github.com/cytosmart-bv/tomni"""

    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def _rescale_intensity(self, img, min_val, max_val):
        output_img = np.clip(img, min_val, max_val)
        if max_val - min_val == 0:
            return (output_img * 255).astype(np.uint8)
        output_img = (output_img - min_val) / (max_val - min_val) * 255
        return output_img.astype(np.uint8)

    def call(self, x: List[np.ndarray], args: List[int]):
        # kernel = kernel_from_parameters(args)
        # img = cv2.morphologyEx(x[0], cv2.MORPH_TOPHAT, kernel, iterations=10)
        kur = np.mean(kurtosis(x[0], fisher=True))
        skew1 = np.mean(skew(x[0]))
        if kur > 1 and skew1 > 1:
            p2, p98 = np.percentile(x[0], (15, 99.5), interpolation="linear")
        else:
            p2, p98 = np.percentile(x[0], (15, 100), interpolation="linear")

        return self._rescale_intensity(x[0], p2, p98)


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


@register(Primitive, "binary_in_range")
class BinaryInRange(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        lower = int(min(args[0], args[1]))
        upper = int(max(args[0], args[1]))
        return cv2.inRange(x[0], lower, upper)


@register(Primitive, "in_range")
class InRange(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        lower = int(min(args[0], args[1]))
        upper = int(max(args[0], args[1]))
        return cv2.bitwise_and(
            x[0],
            x[0],
            mask=cv2.inRange(x[0], lower, upper),
        )


@register(Primitive, "kirsch")
class Kirsch(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        compass_gradients = [
            cv2.filter2D(x[0], ddepth=cv2.CV_32F, kernel=kernel)
            for kernel in KERNEL_KIRSCH_COMPASS
        ]
        return cv2.convertScaleAbs(np.max(compass_gradients, axis=0))


@register(Primitive, "embossing")
class Embossing(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        return cv2.convertScaleAbs(
            cv2.filter2D(x[0], ddepth=cv2.CV_32F, kernel=KERNEL_EMBOSS)
        )


@register(Primitive, "normalize")
class Normalize(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        return cv2.normalize(x[0], None, 0, 255, cv2.NORM_MINMAX)


@register(Primitive, "denoize")
class Denoize(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x, args=None):
        return cv2.fastNlMeansDenoising(x[0], None, h=int(args[0]))


@register(Primitive, "pyr_up")
class PyrUpPrimitive(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        h, w = x[0].shape
        scaled_twice = cv2.pyrUp(x[0])
        return cv2.resize(scaled_twice, (w, h))


@register(Primitive, "pyr_down")
class PyrDown(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        h, w = x[0].shape
        scaled_half = cv2.pyrDown(x[0])
        return cv2.resize(scaled_half, (w, h))


library_opencv = LibraryDefaultOpenCV()
library_opencv.add_by_name("max")
library_opencv.add_by_name("min")
library_opencv.add_by_name("mean")
library_opencv.add_by_name("add")
library_opencv.add_by_name("subtract")
library_opencv.add_by_name("bitwise_not")
library_opencv.add_by_name("bitwise_or")
library_opencv.add_by_name("bitwise_and")
library_opencv.add_by_name("bitwise_and_mask")
library_opencv.add_by_name("bitwise_xor")
library_opencv.add_by_name("sqrt")
library_opencv.add_by_name("pow")
library_opencv.add_by_name("exp")
library_opencv.add_by_name("log")
library_opencv.add_by_name("median_blur")
library_opencv.add_by_name("gaussian_blur")
library_opencv.add_by_name("laplacian")
library_opencv.add_by_name("sobel")
library_opencv.add_by_name("roberts")
library_opencv.add_by_name("canny")
library_opencv.add_by_name("sharpen")
library_opencv.add_by_name("gaussian_diff")
library_opencv.add_by_name("abs_diff")
library_opencv.add_by_name("erode")
library_opencv.add_by_name("dilate")
library_opencv.add_by_name("open")
library_opencv.add_by_name("close")
library_opencv.add_by_name("gradient")
library_opencv.add_by_name("top_hat")
library_opencv.add_by_name("black_hat")
library_opencv.add_by_name("hit_miss")
library_opencv.add_by_name("fill")
library_opencv.add_by_name("rm_small_objects")
library_opencv.add_by_name("rm_small_holes")
library_opencv.add_by_name("binary_threshold")
library_opencv.add_by_name("to_zero_threshold")
library_opencv.add_by_name("binarize")
library_opencv.add_by_name("binary_in_range")
library_opencv.add_by_name("in_range")
library_opencv.add_by_name("fluo_tophat")
library_opencv.add_by_name("kirsch")
# library_opencv.add_by_name("embossing")
library_opencv.add_by_name("normalize")
# library_opencv.add_by_name("denoize")
library_opencv.add_by_name("pyr_up")
library_opencv.add_by_name("pyr_down")


"""
library_opencv.create_primitive("Min", 2, 0, f_min)
library_opencv.create_primitive("Mean", 2, 0, f_mean)
library_opencv.create_primitive("Add", 2, 0, f_add)
library_opencv.create_primitive("Subtract", 2, 0, f_sub)
library_opencv.create_primitive("Bitwise Not", 1, 0, f_bitwise_not)
library_opencv.create_primitive("Bitwise Or", 2, 0, f_bitwise_or)
library_opencv.create_primitive("Bitwise And", 2, 0, f_bitwise_and)
library_opencv.create_primitive("Bitwise And Mask", 2, 0, f_bitwise_and_mask)
library_opencv.create_primitive("Bitwise Xor", 2, 0, f_bitwise_xor)
library_opencv.create_primitive("Square Root", 1, 0, f_sqrt)
library_opencv.create_primitive("Power 2", 1, 0, f_pow)
library_opencv.create_primitive("Exp", 1, 0, f_exp)
library_opencv.create_primitive("Log", 1, 0, f_log)
library_opencv.create_primitive("Median Blur", 1, 1, f_median_blur)
library_opencv.create_primitive("Gaussian Blur", 1, 1, f_gaussian_blur)
library_opencv.create_primitive("Laplacian", 1, 0, f_laplacian)
library_opencv.create_primitive("Sobel", 1, 2, f_sobel)
library_opencv.create_primitive("Roberts", 1, 1, f_roberts)
library_opencv.create_primitive("Canny", 1, 2, f_canny)
library_opencv.create_primitive("Sharpen", 1, 0, f_sharpen)
library_opencv.create_primitive("Gabor", 1, 2, f_gabor)
library_opencv.create_primitive("Subtract Gaussian", 1, 2, f_diff_gaussian)
library_opencv.create_primitive("Absolute Difference", 2, 0, f_absdiff)
library_opencv.create_primitive("Fluo TopHat", 1, 2, f_fluo_tophat)
library_opencv.create_primitive("Relative Difference", 1, 1, f_relative_diff)
library_opencv.create_primitive("Erode", 1, 2, f_erode)
library_opencv.create_primitive("Dilate", 1, 2, f_dilate)
library_opencv.create_primitive("Open", 1, 2, f_open)
library_opencv.create_primitive("Close", 1, 2, f_close)
library_opencv.create_primitive("Morph Gradient", 1, 2, f_morph_gradient)
library_opencv.create_primitive("Morph Tophat", 1, 2, f_morph_tophat)
library_opencv.create_primitive("Morph BlackHat", 1, 2, f_morph_blackhat)
library_opencv.create_primitive("Morph Fill", 1, 0, f_fill)
library_opencv.create_primitive("Remove Small Objects", 1, 1, f_remove_small_objects)
library_opencv.create_primitive("Remove Small Holes", 1, 1, f_remove_small_holes)
library_opencv.create_primitive("Threshold", 1, 2, f_threshold)
library_opencv.create_primitive("Threshold at 1", 1, 1, f_threshold_at_1)
library_opencv.create_primitive("Distance Transform", 1, 1, f_distance_transform)
library_opencv.create_primitive(
    "Distance Transform Threshold", 1, 2, f_distance_transform_thresh
)
library_opencv.create_primitive("Binary In Range", 1, 2, f_bin_inrange)
library_opencv.create_primitive("In Range", 1, 2, f_inrange)

"""


def no_endpoint(x):
    return x


if __name__ == "__main__":
    library = library_opencv
    library.display()
    decoder = SequentialDecoder(2, 30, library)
    with open("decoder.toml", "w") as toml_file:
        toml.dump(decoder.to_toml(), toml_file)
    print(toml.dumps(decoder.to_toml()))

    with open("decoder.toml", "r") as toml_file:
        toml_data = toml.load(toml_file)
        print(toml_data)
