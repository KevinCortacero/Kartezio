from typing import List

import cv2
import numpy as np
from kartezio.components.core import register
from kartezio.components.library import Library, Primitive
from kartezio.thirdparty.kuwahara import kuwahara_filter
from kartezio.types import TypeArray, TypeScalar
from kartezio.vision.common import (
    convolution,
    gradient_magnitude,
    morph_fill,
    threshold_binary,
    threshold_tozero,
)
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
from scipy.stats import kurtosis, skew
from skimage.feature import local_binary_pattern
from skimage.filters import frangi, hessian, meijering, sato
from skimage.morphology import remove_small_holes, remove_small_objects


@register(Library, "opencv_library")
class LibraryDefaultOpenCV(Library):
    def __init__(self):
        super().__init__(TypeArray)


@register(Primitive, "identity")
class Identity(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return x[0]
    

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


@register(Primitive, "add_scalar")
class AddScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.add(x[0], float(x[1]))


@register(Primitive, "subtract")
class Subtract(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.subtract(x[0], x[1])


@register(Primitive, "subtract_scalar")
class SubtractScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.subtract(x[0], float(x[1]))
    

@register(Primitive, "multiply")
class Multiply(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.multiply(x[0], x[1])
    

@register(Primitive, "multiply_scalar")
class MultiplyScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.multiply(x[0], float(x[1]))
    

@register(Primitive, "divide")
class Divide(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.divide(x[0], x[1])
    

@register(Primitive, "divide_scalar")
class DivideScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.divide(x[0], float(x[1]))


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


@register(Primitive, "pow2")
class Pow2(Primitive):
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
        return (cv2.exp((x[0] / 255.0).astype(np.float32)) * 255).astype(np.uint8)


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


@register(Primitive, "median_blur_scalar")
class MedianBlurScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.medianBlur(x[0], correct_ksize(args[1]))


@register(Primitive, "gaussian_blur")
class GaussianBlur(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        ksize = correct_ksize(args[0])
        return cv2.GaussianBlur(x[0], (ksize, ksize), 0)


@register(Primitive, "gaussian_blur_scalar")
class GaussianBlurScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        ksize = correct_ksize(args[1])
        return cv2.GaussianBlur(x[0], (ksize, ksize), 0)


@register(Primitive, "laplacian")
class Laplacian(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.convertScaleAbs(cv2.Laplacian(x[0], cv2.CV_8U, ksize=3))


# future: def f_sobel_laplacian(x, args=None): return cv2.convertScaleAbs(cv2.Laplacian(x[0], cv2.CV_8U, ksize=3))


@register(Primitive, "sobel")
class Sobel(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        gx, gy = cv2.spatialGradient(x[0])
        return gradient_magnitude(gx.astype(np.float32), gy.astype(np.float32))


@register(Primitive, "deriche")
class Deriche(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.alpha = 1.0
        self.omega = 1.0

    def call(self, x: List[np.ndarray], args: List[int]):
        gx = cv2.ximgproc.GradientDericheX(x[0], self.alpha, self.omega)
        gy = cv2.ximgproc.GradientDericheY(x[0], self.alpha, self.omega)
        return gradient_magnitude(gx, gy)


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
        t1 = float(args[0])
        t2 = t1 * 3 if t1 * 3 < 255.0 else 255.0
        return cv2.Canny(x[0], t1, t2)


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


@register(Primitive, "abs_diff2")
class AbsoluteDifference2(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return 255 - cv2.absdiff(x[0], x[1])


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
        return cv2.morphologyEx(x[0], cv2.MORPH_OPEN, kernel_from_parameters(args))


@register(Primitive, "close")
class Close(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_CLOSE, kernel_from_parameters(args))


@register(Primitive, "gradient")
class Gradient(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_GRADIENT, kernel_from_parameters(args))


@register(Primitive, "morph_gradient")
class MorphGradient(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_GRADIENT, kernel_from_parameters(args))


@register(Primitive, "top_hat")
class TopHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_TOPHAT, kernel_from_parameters(args))


@register(Primitive, "morph_tophat")
class MorphTopHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_TOPHAT, kernel_from_parameters(args))


@register(Primitive, "black_hat")
class BlackHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_BLACKHAT, kernel_from_parameters(args))


@register(Primitive, "morph_blackhat")
class MorphBlackHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_BLACKHAT, kernel_from_parameters(args))


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


@register(Primitive, "fill_holes")
class FillHoles(Primitive):
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


@register(Primitive, "remove_small_objects")
class RemSmallObjects(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return remove_small_objects(x[0] > 0, args[0]).astype(np.uint8)


@register(Primitive, "remove_small_holes")
class RemSmallHoles(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return remove_small_holes(x[0] > 0, args[0]).astype(np.uint8)


@register(Primitive, "threshold")
class Threshold(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return threshold_tozero(x[0], args[0])
    

@register(Primitive, "threshold_scalar")
class ThresholdScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return threshold_tozero(x[0], x[1])


@register(Primitive, "kuwahara")
class Kuwahara(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return kuwahara_filter(x[0], correct_ksize(args[0]))


@register(Primitive, "distance_transform")
class DistanceTransform(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x, args=None):
        return cv2.normalize(
            cv2.distanceTransform(x[0].copy(), cv2.DIST_L2, 3),
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            cv2.CV_8U,
        )


@register(Primitive, "distance_transform_and_thresh")
class DistanceTransformAndThresh(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x, args=None):
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


@register(Primitive, "inrange_bin")
class BinaryInRange2(Primitive):
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


@register(Primitive, "inrange")
class InRange2(Primitive):
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


@register(Primitive, "meijiring")
class Meijiring(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return cv2.convertScaleAbs(meijering(x[0], sigmas=self.sigmas) * 255)


@register(Primitive, "sato")
class Sato(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return cv2.convertScaleAbs(sato(x[0], sigmas=self.sigmas) * 255)


@register(Primitive, "frangi")
class Frangi(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return cv2.convertScaleAbs(frangi(x[0], sigmas=self.sigmas) * 255)


@register(Primitive, "hessian")
class Hessian(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return cv2.convertScaleAbs(hessian(x[0], sigmas=self.sigmas) * 255)


@register(Primitive, "gabor")
class GaborFilter(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)
        self.ksize = 11

    def call(self, x, args=None):
        gabor_k = gabor_kernel(self.ksize, args[0], args[1])
        return cv2.filter2D(x[0], -1, gabor_k)


@register(Primitive, "gabor_11")
class Gabor11(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)
        self.ksize = 11
        self.sigma = 2.0
        self.angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    def call(self, x, args=None):
        lambd = args[0] / 255.0
        gamma = args[1] // 16
        gabored = []
        for angle in self.angles:
            gabor_kernel = cv2.getGaborKernel(
                (self.ksize, self.ksize),
                self.sigma,
                angle,
                lambd,
                gamma,
                psi=0,
            )
            gabored.append(cv2.filter2D(x[0], -1, gabor_kernel))
        return np.max(gabored, axis=0)


@register(Primitive, "gabor_7")
class Gabor7(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)
        self.ksize = 7
        self.sigma = 1.0
        self.angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    def call(self, x, args=None):
        lambd = args[0] / 255.0
        gamma = args[1] // 16
        gabored = []
        for angle in self.angles:
            gabor_kernel = cv2.getGaborKernel(
                (self.ksize, self.ksize),
                self.sigma,
                angle,
                lambd,
                gamma,
                psi=0,
            )
            gabored.append(cv2.filter2D(x[0], -1, gabor_kernel))
        return np.max(gabored, axis=0)


@register(Primitive, "gabor_3")
class Gabor3(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)
        self.ksize = 3
        self.sigma = 1.0
        self.angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    def call(self, x, args=None):
        lambd = args[0] / 255.0
        gamma = args[1] // 16
        gabored = []
        for angle in self.angles:
            gabor_kernel = cv2.getGaborKernel(
                (self.ksize, self.ksize),
                self.sigma,
                angle,
                lambd,
                gamma,
                psi=0,
            )
            gabored.append(cv2.filter2D(x[0], -1, gabor_kernel))
        return np.max(gabored, axis=0)


@register(Primitive, "local_binary_pattern")
class LocalBinaryPattern(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x, args=None):
        return local_binary_pattern(x[0], 8, args[0] // 16, method="uniform").astype(
            np.uint8
        )


@register(Primitive, "laplacian_of_gaussian")
class LaplacianOfGaussian(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x, args=None):
        image = x[0]
        sigma = 2.0
        size = correct_ksize(args[0])

        x, y = np.meshgrid(
            np.arange(-size // 2 + 1, size // 2 + 1),
            np.arange(-size // 2 + 1, size // 2 + 1),
        )
        kernel = (
            -(1 / (np.pi * sigma**4))
            * (1 - ((x**2 + y**2) / (2 * sigma**2)))
            * np.exp(-(x**2 + y**2) / (2 * sigma**2))
        )
        kernel = kernel / np.sum(np.abs(kernel))

        return cv2.convertScaleAbs(cv2.filter2D(image, -1, kernel))


def create_array_lib(use_scalars=False):
    library_opencv = LibraryDefaultOpenCV()
    library_opencv.add_by_name("identity")
    library_opencv.add_by_name("max")
    library_opencv.add_by_name("min")
    library_opencv.add_by_name("mean")
    if use_scalars:
        library_opencv.add_by_name("add_scalar")
        library_opencv.add_by_name("subtract_scalar")
        library_opencv.add_by_name("multiply_scalar")
        library_opencv.add_by_name("divide_scalar")
        library_opencv.add_by_name("median_blur_scalar")
        library_opencv.add_by_name("gaussian_blur_scalar")
        library_opencv.add_by_name("threshold_scalar")
    else:
        library_opencv.add_by_name("add")
        library_opencv.add_by_name("subtract")
        library_opencv.add_by_name("multiply")
        library_opencv.add_by_name("divide")
        library_opencv.add_by_name("median_blur")
        library_opencv.add_by_name("gaussian_blur")
        library_opencv.add_by_name("threshold")
    library_opencv.add_by_name("bitwise_not")
    library_opencv.add_by_name("bitwise_or")
    library_opencv.add_by_name("bitwise_and")
    library_opencv.add_by_name("bitwise_and_mask")
    library_opencv.add_by_name("bitwise_xor")
    library_opencv.add_by_name("sqrt")
    library_opencv.add_by_name("pow")
    library_opencv.add_by_name("exp")
    library_opencv.add_by_name("log")
    library_opencv.add_by_name("laplacian")
    library_opencv.add_by_name("sobel")
    library_opencv.add_by_name("deriche")
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
    library_opencv.add_by_name("binary_in_range")
    library_opencv.add_by_name("fill")
    library_opencv.add_by_name("rm_small_objects")
    library_opencv.add_by_name("rm_small_holes")
    library_opencv.add_by_name("in_range")
    library_opencv.add_by_name("pyr_up")
    library_opencv.add_by_name("pyr_down")
    library_opencv.add_by_name("kirsch")
    library_opencv.add_by_name("embossing")
    library_opencv.add_by_name("normalize")
    library_opencv.add_by_name("denoize")
    library_opencv.add_by_name("local_binary_pattern")
    library_opencv.add_by_name("gabor_3")
    library_opencv.add_by_name("gabor_7")
    library_opencv.add_by_name("gabor_11")
    library_opencv.add_by_name("laplacian_of_gaussian")
    library_opencv.add_by_name("meijiring")
    library_opencv.add_by_name("sato")
    library_opencv.add_by_name("frangi")
    library_opencv.add_by_name("hessian")
    library_opencv.add_by_name("kuwahara")
    return library_opencv
