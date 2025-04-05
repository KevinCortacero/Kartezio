from typing import List

import cv2
import numpy as np
import pywt
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from skimage.feature import local_binary_pattern
from skimage.filters import (
    farid,
    frangi,
    hessian,
    meijering,
    prewitt,
    roberts,
    sato,
    scharr,
    sobel,
)
from skimage.morphology import remove_small_holes, remove_small_objects

from kartezio.core.components import (
    Library,
    Primitive,
    register,
    to_f32,
    to_u8,
)
from kartezio.thirdparty.kuwahara import kuwahara_filter
from kartezio.types import TypeArray, TypeScalar
from kartezio.vision.common import (
    convolution,
    gradient_magnitude,
    morph_fill,
    threshold_tozero,
)
from kartezio.vision.dtype import u8, u16, uf32
from kartezio.vision.kernel import (
    KERNEL_EMBOSS,
    KERNEL_KIRSCH_COMPASS,
    KERNEL_ROBERTS_X,
    KERNEL_ROBERTS_Y,
    SHARPEN_KERNEL,
    disk_kernel,
    get_hitmiss_kernel,
    get_ksize_from_params,
)


class ImageFilter(Primitive):
    def __init__(self, n_parameters: int = 0):
        super().__init__([TypeArray], TypeArray, n_parameters)


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
        return x[0] + x[1]


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
        return x[0] - x[1]


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
        return x[0] * x[1]


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
        return x[0] / (x[1] + 1e-4)


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
        filtered = np.zeros_like(x[0])
        filtered[x[1] > 0.0] = x[0][x[1] > 0.0]
        return filtered


@register(Primitive)
class BitwiseXor(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.bitwise_xor(x[0], x[1])


@register(Primitive)
class Sqrt(ImageFilter):
    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.sqrt(np.clip(x[0], 0.0, None))


@register(Primitive)
class Pow(ImageFilter):
    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.pow(x[0], 2)


@register(Primitive)
class Exp(ImageFilter):
    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.exp(x[0])


@register(Primitive)
class Log(ImageFilter):
    def call(self, x: List[np.ndarray], args: List[int]):
        return np.log1p(x[0])


@register(Primitive)
class MedianBlur(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return to_f32(
            cv2.medianBlur(to_u8(x[0]), ksize=get_ksize_from_params(args[0]))
        )


@register(Primitive)
class MedianBlurScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.medianBlur(x[0], ksize=get_ksize_from_params(args[0]))


@register(Primitive)
class GaussianBlur(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        ksize = get_ksize_from_params(args[0])
        return cv2.GaussianBlur(x[0], (ksize, ksize), 0)


@register(Primitive)
class GaussianBlurScalar(Primitive):
    def __init__(self):
        super().__init__([TypeArray, TypeScalar], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        ksize = get_ksize_from_params(x[1])
        return cv2.GaussianBlur(x[0], (ksize, ksize), 0)


@register(Primitive)
class Laplacian(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.Laplacian(
            x[0], cv2.CV_8U, ksize=get_ksize_from_params(args[0])
        )


@register(Primitive)
class SobelScipy(ImageFilter):
    def __init__(self):
        super().__init__(n_parameters=1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return sobel(x[0], mode="nearest")


@register(Primitive)
class FaridScipy(ImageFilter):
    def __init__(self):
        super().__init__(n_parameters=1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return farid(x[0], mode="nearest")


@register(Primitive)
class ScharrScipy(ImageFilter):
    def __init__(self):
        super().__init__(n_parameters=1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return scharr(x[0], mode="nearest")


@register(Primitive)
class AcceleratedSegmentTest(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        """
        Detect corners using the AGAST corner detector in OpenCV.

        Parameters:
            image (ndarray): Input image (grayscale).
            threshold (int): Threshold for corner detection. Higher values mean fewer corners.

        Returns:
            corners (ndarray): Detected corner points in the image.
        """
        # Create the AGAST feature detector
        detector = cv2.AgastFeatureDetector_create()

        # Detect corners in the image
        keypoints = detector.detect(x[0], threshold=args[0] / 255.0)
        y = np.zeros_like(x[0])
        y = cv2.drawKeypoints(
            y, keypoints, None, 1.0, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return y[:, :, 0]


@register(Primitive)
class RobertsScipy(ImageFilter):
    def call(self, x: List[np.ndarray], args: List[int]):
        return roberts(x[0])


@register(Primitive)
class LaplacianScipy(ImageFilter):
    def call(self, x: List[np.ndarray], args: List[int]):
        from skimage.filters import laplace

        ksize = get_ksize_from_params(args[0])
        return laplace(x[0], ksize=ksize)


@register(Primitive)
class PrewittScipy(ImageFilter):
    def call(self, x: List[np.ndarray], args: List[int]):
        from skimage.filters import prewitt

        return prewitt(x[0], mode="nearest")


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
        ksize = get_ksize_from_params(args[0])
        image = x[0] / 255.0
        offset = args[1] / 255.0
        return image - cv2.GaussianBlur(image, (ksize, ksize), 0) + offset


@register(Primitive)
class AbsDiff(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.absdiff(x[0], x[1])


@register(Primitive)
class Erode(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.erode(x[0], disk_kernel(args[0]))


@register(Primitive)
class Dilate(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.dilate(x[0], disk_kernel(args[0]))


@register(Primitive)
class Open(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_OPEN, disk_kernel(args[0]))


@register(Primitive)
class Close(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_CLOSE, disk_kernel(args[0]))


@register(Primitive)
class MorphGradient(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_GRADIENT, disk_kernel(args[0]))


@register(Primitive)
class MorphTopHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_TOPHAT, disk_kernel(args[0]))


@register(Primitive)
class MorphBlackHat(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return cv2.morphologyEx(x[0], cv2.MORPH_BLACKHAT, disk_kernel(args[0]))


@register(Primitive)
class HitMiss(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        ksize = get_ksize_from_params(args[0])
        kernel = get_hitmiss_kernel(ksize)
        return to_f32(cv2.morphologyEx(to_u8(x[0]), cv2.MORPH_HITMISS, kernel))


@register(Primitive)
class Fill(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return morph_fill(to_u8(x[0]))


@register(Primitive)
class RmSmallObjects(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)
        self.scale = 10

    def call(self, x: List[np.ndarray], args: List[int]):
        mask = remove_small_objects(x[0] > 0.0, args[0] * self.scale)
        filtered = np.zeros_like(x[0])
        filtered[mask == 0] = 0.0
        return filtered


@register(Primitive)
class RmSmallHoles(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)
        self.scale = 10

    def call(self, x: List[np.ndarray], args: List[int]):
        mask = remove_small_holes(x[0] > 0.0, args[0] * self.scale)
        filtered = np.zeros_like(x[0])
        filtered[mask == 0] = 0.0
        return filtered


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
        return kuwahara_filter(x[0], ksize=get_ksize_from_params(args[0]))


@register(Primitive)
class DistanceTransform(ImageFilter):
    def __init__(self):
        super().__init__(1)

    def call(self, x, args=None):
        return self.to_uint8(cv2.distanceTransform(x[0], cv2.DIST_L2, 3))


@register(Primitive)
class DTScipy(ImageFilter):
    def __init__(self):
        super().__init__(1)

    def call(self, x, args=None):

        # Compute the distance transform using the Euclidean distance
        distance = distance_transform_edt(x[0])
        # Normalize the distance transform to the range [0, 1]
        m = np.max(distance)
        if m == 0.0:
            return np.zeros_like(distance, dtype=np.float32)
        return (distance / m).astype(np.float32)


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
        return 1.0 - x[0]


@register(Primitive)
class InverseNonZero(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x: List[np.ndarray], args: List[int] = None):
        inverse = 1.0 - x[0]
        inverse[inverse == 1.0] = 0.0
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
        return np.max(compass_gradients, axis=0)


@register(Primitive)
class Embossing(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        return cv2.filter2D(x[0], ddepth=cv2.CV_32F, kernel=KERNEL_EMBOSS)


@register(Primitive)
class Normalize(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        return cv2.normalize(x[0], None, 0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)


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
        return cv2.resize(
            scaled_twice, (w, h), interpolation=cv2.INTER_NEAREST
        )


@register(Primitive)
class PyrDown(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)

    def call(self, x, args=None):
        h, w = x[0].shape
        scaled_half = cv2.pyrDown(x[0])
        return cv2.resize(scaled_half, (w, h), interpolation=cv2.INTER_NEAREST)


@register(Primitive)
class Meijiring(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return meijering(x[0], sigmas=self.sigmas)


@register(Primitive)
class Sato(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return sato(x[0], sigmas=self.sigmas)


@register(Primitive)
class Frangi(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return frangi(x[0], sigmas=self.sigmas)


@register(Primitive)
class Hessian(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 0)
        self.sigmas = [1.0]

    def call(self, x, args=None):
        return hessian(x[0], sigmas=self.sigmas)


class Gabor(Primitive):
    def __init__(self, ksize):
        super().__init__([TypeArray], TypeArray, 2)
        self.ksize = ksize
        self.sigma = 1.0
        self.psi = 0.0
        self.angles = np.linspace(0, 1, 5)[:4] * np.pi

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
                self.psi,
            )
            gabored.append(
                cv2.filter2D(x[0], ddepth=cv2.CV_32F, kernel=gabor_kernel)
            )
        return np.max(gabored, axis=0)


@register(Primitive)
class Gabor3(Gabor):
    def __init__(self):
        super().__init__(3)


@register(Primitive)
class Gabor5(Gabor):
    def __init__(self):
        super().__init__(5)


@register(Primitive)
class Gabor7(Gabor):
    def __init__(self):
        super().__init__(7)


@register(Primitive)
class Gabor9(Gabor):
    def __init__(self):
        super().__init__(9)


@register(Primitive)
class LocalBinaryPattern(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 1)

    def call(self, x, args=None):
        return local_binary_pattern(
            x[0], 8, args[0] // 16, method="uniform"
        ).astype(np.uint8)


class WaveletDenoising(Primitive):
    def __init__(self, wavelet):
        super().__init__([TypeArray], TypeArray, 1)
        self.wavelet = wavelet

    def call(self, x, args=None):
        coeffs = pywt.wavedec2(
            x[0].astype(np.float32), self.wavelet, level=args[0] // 64
        )
        coeffs[1:] = [
            pywt.threshold(i, value=0.1 * np.max(i)) for i in coeffs[1:]
        ]
        denoised_data = pywt.waverec(coeffs, self.wavelet)
        return denoised_data.astype(np.uint8)


@register(Primitive)
class WaveletDB1(WaveletDenoising):
    def __init__(self):
        super().__init__("db1")


@register(Primitive)
class WaveletHaar(WaveletDenoising):
    def __init__(self):
        super().__init__("haar")


@register(Primitive)
class SavitzkyGolay(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeArray, 2)

    def call(self, x, args=None):
        ksize = get_ksize_from_params(args[0])
        return savgol_filter(x[0], window_length=ksize, polyorder=2)


class ArrayLib(Library):
    def __init__(self):
        super().__init__(TypeArray)
        self.add_primitive(Identity())

    def add_logical(self):
        self.add_primitive(BitwiseNot())
        self.add_primitive(BitwiseOr())
        self.add_primitive(BitwiseAnd())
        self.add_primitive(BitwiseAndMask())
        self.add_primitive(BitwiseXor())

    def add_arithmetic(self, use_scalars=False):
        self.add_primitive(Add())
        self.add_primitive(Subtract())
        self.add_primitive(Multiply())
        self.add_primitive(Divide())
        if use_scalars:
            self.add_primitive(AddScalar())
            self.add_primitive(SubtractScalar())
            self.add_primitive(MultiplyScalar())
            self.add_primitive(DivideScalar())

    def add_comparative(self):
        self.add_primitive(Max())
        self.add_primitive(Min())
        self.add_primitive(Mean())
        self.add_primitive(AbsDiff())

    def add_morphological(self):
        self.add_primitive(Erode())
        self.add_primitive(Dilate())
        self.add_primitive(Open())
        self.add_primitive(Close())
        self.add_primitive(MorphGradient())
        self.add_primitive(MorphTopHat())
        self.add_primitive(MorphBlackHat())
        self.add_primitive(HitMiss())
        # self.add_primitive(Fill())
        self.add_primitive(RmSmallObjects())
        self.add_primitive(RmSmallHoles())
        self.add_primitive(DTScipy())

    def add_blurring(self, use_scalars=False):
        if use_scalars:
            self.add_primitive(MedianBlurScalar())
            self.add_primitive(GaussianBlurScalar())
        else:
            self.add_primitive(MedianBlur())
            self.add_primitive(GaussianBlur())
        self.add_primitive(Sharpen())

    def add_edge_detection(self):
        self.add_primitive(LaplacianScipy())
        self.add_primitive(SobelScipy())
        self.add_primitive(ScharrScipy())
        self.add_primitive(FaridScipy())
        self.add_primitive(RobertsScipy())
        self.add_primitive(PrewittScipy())
        self.add_primitive(Kirsch())
        self.add_primitive(Embossing())

    def add_gabor_filters(self):
        self.add_primitive(Gabor3())
        self.add_primitive(Gabor5())
        self.add_primitive(Gabor7())
        self.add_primitive(Gabor9())

    def add_histogram_primitives(self, use_scalars=False):
        if use_scalars:
            self.add_primitive(ThresholdScalar())
        else:
            self.add_primitive(Threshold())
        self.add_primitive(InRange())
        self.add_primitive(Normalize())
        self.add_primitive(Sqrt())
        self.add_primitive(Pow())
        self.add_primitive(Exp())
        self.add_primitive(Log())
        self.add_primitive(Inverse())
        self.add_primitive(InverseNonZero())

    def add_denoising_filters(self):
        # self.add_primitive(WaveletDB1())
        # self.add_primitive(WaveletHaar())
        self.add_primitive(SavitzkyGolay())
        self.add_primitive(PyrUp())
        self.add_primitive(PyrDown())

    def add_advanced_filters(self):
        self.add_primitive(Meijiring())
        self.add_primitive(Sato())
        self.add_primitive(Frangi())
        self.add_primitive(Hessian())
        self.add_primitive(Kuwahara())
