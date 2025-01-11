from typing import List

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import frangi, hessian, meijering, sato
from skimage.morphology import remove_small_holes, remove_small_objects

from kartezio.core.components import Library, Primitive, register
from kartezio.thirdparty.kuwahara import kuwahara_filter
from kartezio.types import TypeArray, TypeScalar
from kartezio.vision.common import (
    convolution,
    gradient_magnitude,
    morph_fill,
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


@register(Primitive)
class Identity(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return x[0]


@register(Primitive)
class Max(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.max(x[0], x[1])


@register(Primitive)
class Min(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.min(x[0], x[1])


@register(Primitive)
class Mean(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.addWeighted(x[0], 0.5, x[1], 0.5, 0)


@register(Primitive)
class Add(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.add(x[0], x[1])


@register(Primitive)
class AddScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.add(x[0], float(x[1]))


@register(Primitive)
class Subtract(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.subtract(x[0], x[1])


@register(Primitive)
class SubtractScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.subtract(x[0], float(x[1]))


@register(Primitive)
class Multiply(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.multiply(x[0], x[1])


@register(Primitive)
class MultiplyScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.multiply(x[0], float(x[1]))


@register(Primitive)
class Divide(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.divide(x[0], x[1])


@register(Primitive)
class DivideScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.divide(x[0], float(x[1]))


@register(Primitive)
class BitwiseNot(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_not(x[0])


@register(Primitive)
class BitwiseOr(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_or(x[0], x[1])


@register(Primitive)
class BitwiseAnd(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_and(x[0], x[1])


@register(Primitive)
class BitwiseAndMask(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_and(x[0], x[0], mask=x[1])


@register(Primitive)
class BitwiseXor(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_xor(x[0], x[1])


@register(Primitive)
class Sqrt(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.convertScaleAbs(cv2.sqrt(x[0].astype(np.float32)))


@register(Primitive)
class Pow(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.convertScaleAbs(cv2.pow(x[0], 2))


@register(Primitive)
class Pow2(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.convertScaleAbs(cv2.pow(x[0], 2))


@register(Primitive)
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


@register(Primitive)
class Log(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.convertScaleAbs(np.log1p(x[0].astype(np.float32)))


@register(Primitive)
class MedianBlur(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.medianBlur(x[0], correct_ksize(args[0]))


@register(Primitive)
class MedianBlurScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.medianBlur(x[0], correct_ksize(args[1]))


@register(Primitive)
class GaussianBlur(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        ksize = correct_ksize(args[0])
        return cv2.GaussianBlur(x[0], (ksize, ksize), 0)


@register(Primitive)
class GaussianBlurScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        ksize = correct_ksize(args[1])
        return cv2.GaussianBlur(x[0], (ksize, ksize), 0)


@register(Primitive)
class Laplacian(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.convertScaleAbs(cv2.Laplacian(x[0], cv2.CV_8U, ksize=3))


# future: def f_sobel_laplacian(x, args=None): return cv2.convertScaleAbs(cv2.Laplacian(x[0], cv2.CV_8U, ksize=3))


@register(Primitive)
class Sobel(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        gx, gy = cv2.spatialGradient(x[0])
        return gradient_magnitude(gx.astype(np.float32), gy.astype(np.float32))


@register(Primitive)
class Roberts(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        gx = convolution(x[0], KERNEL_ROBERTS_X)
        gy = convolution(x[0], KERNEL_ROBERTS_Y)
        return gradient_magnitude(gx.astype(np.float32), gy.astype(np.float32))


@register(Primitive)
class Canny(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        t1 = float(args[0])
        t2 = t1 * 3 if t1 * 3 < 255.0 else 255.0
        return cv2.Canny(x[0], t1, t2)


@register(Primitive)
class Sharpen(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return convolution(x[0], SHARPEN_KERNEL)


@register(Primitive)
class GaussianDiff(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        ksize = correct_ksize(args[0])
        image = x[0].astype(np.int16)
        return cv2.convertScaleAbs(
            image - cv2.GaussianBlur(image, (ksize, ksize), 0) + args[1]
        )


@register(Primitive)
class AbsDiff(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.absdiff(x[0], x[1])


@register(Primitive)
class AbsoluteDifference2(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return 255 - cv2.absdiff(x[0], x[1])


@register(Primitive)
class Erode(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.erode(x[0], kernel_from_parameters(args))


@register(Primitive)
class Dilate(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.dilate(x[0], kernel_from_parameters(args))


@register(Primitive)
class Open(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_OPEN, kernel_from_parameters(args)
        )


@register(Primitive)
class Close(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_CLOSE, kernel_from_parameters(args)
        )


@register(Primitive)
class Gradient(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_GRADIENT, kernel_from_parameters(args)
        )


@register(Primitive)
class MorphGradient(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_GRADIENT, kernel_from_parameters(args)
        )


@register(Primitive)
class TopHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_TOPHAT, kernel_from_parameters(args)
        )


@register(Primitive)
class MorphTopHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_TOPHAT, kernel_from_parameters(args)
        )


@register(Primitive)
class BlackHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_BLACKHAT, kernel_from_parameters(args)
        )


@register(Primitive)
class MorphBlackHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(
            x[0], cv2.MORPH_BLACKHAT, kernel_from_parameters(args)
        )


@register(Primitive)
class HitMiss(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_HITMISS, HITMISS_KERNEL)


@register(Primitive)
class Fill(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return morph_fill(x[0])


@register(Primitive)
class RmSmallObjects(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return remove_small_objects(x[0] > 0, args[0]).astype(np.uint8)


@register(Primitive)
class RmSmallHoles(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return remove_small_holes(x[0] > 0, args[0]).astype(np.uint8)


@register(Primitive)
class RemSmallObjects(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return remove_small_objects(x[0] > 0, args[0]).astype(np.uint8)


@register(Primitive)
class RemSmallHoles(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return remove_small_holes(x[0] > 0, args[0]).astype(np.uint8)


@register(Primitive)
class Threshold(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return threshold_tozero(x[0], args[0])


@register(Primitive)
class ThresholdScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return threshold_tozero(x[0], x[1])


@register(Primitive)
class Kuwahara(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return kuwahara_filter(x[0], correct_ksize(args[0]))


@register(Primitive)
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


@register(Primitive)
class BinaryInRange(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x: List[np.ndarray], args: List[int]):
        lower = int(min(args[0], args[1]))
        upper = int(max(args[0], args[1]))
        return cv2.inRange(x[0], lower, upper)


@register(Primitive)
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


@register(Primitive)
class Inverse(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int] = None):
        return 255 - x[0]


@register(Primitive)
class InverseNonZero(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int] = None):
        inverse = 255 - x[0]
        inverse[inverse == 255] = 0
        return inverse


@register(Primitive)
class Kirsch(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        compass_gradients = [
            cv2.filter2D(x[0], ddepth=cv2.CV_32F, kernel=kernel)
            for kernel in KERNEL_KIRSCH_COMPASS
        ]
        return cv2.convertScaleAbs(np.max(compass_gradients, axis=0))


@register(Primitive)
class Embossing(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        return cv2.convertScaleAbs(
            cv2.filter2D(x[0], ddepth=cv2.CV_32F, kernel=KERNEL_EMBOSS)
        )


@register(Primitive)
class Normalize(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        return cv2.normalize(x[0], None, 0, 255, cv2.NORM_MINMAX)


@register(Primitive)
class Denoize(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x, args=None):
        return cv2.fastNlMeansDenoising(x[0], None, h=int(args[0]))


@register(Primitive)
class PyrUp(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        h, w = x[0].shape
        scaled_twice = cv2.pyrUp(x[0])
        return cv2.resize(scaled_twice, (w, h))


@register(Primitive)
class PyrDown(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        h, w = x[0].shape
        scaled_half = cv2.pyrDown(x[0])
        return cv2.resize(scaled_half, (w, h))


@register(Primitive)
class Meijiring(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return cv2.convertScaleAbs(meijering(x[0], sigmas=self.sigmas) * 255)


@register(Primitive)
class Sato(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return cv2.convertScaleAbs(sato(x[0], sigmas=self.sigmas) * 255)


@register(Primitive)
class Frangi(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return cv2.convertScaleAbs(frangi(x[0], sigmas=self.sigmas) * 255)


@register(Primitive)
class Hessian(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return cv2.convertScaleAbs(hessian(x[0], sigmas=self.sigmas) * 255)


@register(Primitive)
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


@register(Primitive)
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


@register(Primitive)
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


@register(Primitive)
class LocalBinaryPattern(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x, args=None):
        return local_binary_pattern(
            x[0], 8, args[0] // 16, method="uniform"
        ).astype(np.uint8)


@register(Primitive)
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
    library_opencv = Library(TypeArray)
    library_opencv.add_primitive(Identity())
    library_opencv.add_primitive(Max())
    library_opencv.add_primitive(Min())
    library_opencv.add_primitive(Mean())
    library_opencv.add_primitive(Inverse())
    library_opencv.add_primitive(InverseNonZero())
    if use_scalars:
        library_opencv.add_primitive(AddScalar())
        library_opencv.add_primitive(SubtractScalar())
        library_opencv.add_primitive(MultiplyScalar())
        library_opencv.add_primitive(DivideScalar())
        library_opencv.add_primitive(MedianBlurScalar())
        library_opencv.add_primitive(GaussianBlurScalar())
        library_opencv.add_primitive(ThresholdScalar())
    else:
        library_opencv.add_primitive(Add())
        library_opencv.add_primitive(Subtract())
        library_opencv.add_primitive(Multiply())
        library_opencv.add_primitive(Divide())
        library_opencv.add_primitive(MedianBlur())
        library_opencv.add_primitive(GaussianBlur())
        library_opencv.add_primitive(Threshold())
    library_opencv.add_primitive(BitwiseNot())
    library_opencv.add_primitive(BitwiseOr())
    library_opencv.add_primitive(BitwiseAnd())
    library_opencv.add_primitive(BitwiseAndMask())
    library_opencv.add_primitive(BitwiseXor())
    library_opencv.add_primitive(Sqrt())
    library_opencv.add_primitive(Pow())
    library_opencv.add_primitive(Exp())
    library_opencv.add_primitive(Log())
    library_opencv.add_primitive(Laplacian())
    library_opencv.add_primitive(Sobel())
    library_opencv.add_primitive(Roberts())
    library_opencv.add_primitive(Canny())
    library_opencv.add_primitive(Sharpen())
    library_opencv.add_primitive(GaussianDiff())
    library_opencv.add_primitive(AbsDiff())
    library_opencv.add_primitive(Erode())
    library_opencv.add_primitive(Dilate())
    library_opencv.add_primitive(Open())
    library_opencv.add_primitive(Close())
    library_opencv.add_primitive(Gradient())
    library_opencv.add_primitive(TopHat())
    library_opencv.add_primitive(BlackHat())
    library_opencv.add_primitive(HitMiss())
    library_opencv.add_primitive(Fill())
    library_opencv.add_primitive(RmSmallObjects())
    library_opencv.add_primitive(RmSmallHoles())
    library_opencv.add_primitive(InRange())
    library_opencv.add_primitive(BinaryInRange())
    library_opencv.add_primitive(PyrUp())
    library_opencv.add_primitive(PyrDown())
    library_opencv.add_primitive(Kuwahara())
    library_opencv.add_primitive(DistanceTransform())
    library_opencv.add_primitive(Kirsch())
    library_opencv.add_primitive(Embossing())
    library_opencv.add_primitive(Normalize())
    library_opencv.add_primitive(Denoize())
    library_opencv.add_primitive(LocalBinaryPattern())
    library_opencv.add_primitive(Gabor3())
    library_opencv.add_primitive(Gabor7())
    library_opencv.add_primitive(Gabor11())
    library_opencv.add_primitive(LaplacianOfGaussian())
    library_opencv.add_primitive(Meijiring())
    library_opencv.add_primitive(Sato())
    library_opencv.add_primitive(Frangi())
    library_opencv.add_primitive(Hessian())
    return library_opencv
