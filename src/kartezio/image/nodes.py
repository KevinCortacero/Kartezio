from abc import ABC

import cv2
import numpy as np
from numena.image.basics import image_ew_max, image_ew_mean, image_ew_min
from numena.image.morphology import morph_fill
from numena.image.threshold import threshold_binary, threshold_tozero
from scipy.stats import kurtosis, skew
from skimage.morphology import remove_small_holes, remove_small_objects

from kartezio.image.kernel import (
    ROBERT_CROSS_H_KERNEL,
    ROBERT_CROSS_V_KERNEL,
    SHARPEN_KERNEL,
    correct_ksize,
    gabor_kernel,
    kernel_from_parameters,
)
from kartezio.model.components import KartezioNode, ExportableNode
from kartezio.model.registry import registry


def register_nodes():
    """Force decorators to wrap Nodes"""
    print(f"[Kartezio - INFO] -  {len(registry.nodes.list())} nodes registered.")


class NodeImageProcessing(KartezioNode, ABC):
    def _to_json_kwargs(self) -> dict:
        return {}


@registry.nodes.add("max")
class Max(NodeImageProcessing):
    def __init__(self):
        super().__init__("max", "MAX", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return image_ew_max(x[0], x[1])


@registry.nodes.add("min")
class Min(NodeImageProcessing):
    def __init__(self):
        super().__init__("min", "MIN", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return image_ew_min(x[0], x[1])


@registry.nodes.add("mean")
class Mean(NodeImageProcessing):
    def __init__(self):
        super().__init__("mean", "MEAN", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return image_ew_mean(x[0], x[1])


@registry.nodes.add("add")
class Add(ExportableNode):
    def __init__(self):
        super().__init__("add", "ADD", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.add(x[0], x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.add({input_names[0]}, {input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::add({input_names[0]}, {input_names[1]}, {output_name});"


@registry.nodes.add("subtract")
class Subtract(ExportableNode):
    def __init__(self):
        super().__init__("subtract", "SUB", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.subtract(x[0], x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.subtract({input_names[0]}, {input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::subtract({input_names[0]}, {input_names[1]}, {output_name});"


@registry.nodes.add("bitwise_not")
class BitwiseNot(ExportableNode):
    def __init__(self):
        super().__init__("bitwise_not", "NOT", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_not(x[0])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.bitwise_not({input_names[0]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::bitwise_not({input_names[0]}, {output_name});"


@registry.nodes.add("bitwise_or")
class BitwiseOr(ExportableNode):
    def __init__(self):
        super().__init__("bitwise_or", "BOR", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_or(x[0], x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.bitwise_or({input_names[0]}, {input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::bitwise_or({input_names[0]}, {input_names[1]}, {output_name});"


@registry.nodes.add("bitwise_and")
class BitwiseAnd(ExportableNode):
    def __init__(self):
        super().__init__("bitwise_and", "BAND", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_and(x[0], x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.bitwise_and({input_names[0]}, {input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::bitwise_and({input_names[0]}, {input_names[1]}, {output_name});"


@registry.nodes.add("bitwise_and_mask")
class BitwiseAndMask(ExportableNode):
    def __init__(self):
        super().__init__("bitwise_and_mask", "ANDM", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_and(x[0], x[0], mask=x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.bitwise_and({input_names[0]}, {input_names[0]}, mask={input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::bitwise_and({input_names[0]}, {input_names[0]}, {output_name}, {input_names[1]});"


@registry.nodes.add("bitwise_xor")
class BitwiseXor(ExportableNode):
    def __init__(self):
        super().__init__("bitwise_xor", "BXOR", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.bitwise_xor(x[0], x[1])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.bitwise_xor({input_names[0]}, {input_names[1]})"

    def to_cpp(self, input_names, p, output_name):
        return "cv::bitwise_xor({input_names[0]}, {input_names[1]}, {output_name});"


@registry.nodes.add("sqrt")
class SquareRoot(NodeImageProcessing):
    def __init__(self):
        super().__init__("sqrt", "SQRT", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return (cv2.sqrt((x[0] / 255.0).astype(np.float32)) * 255).astype(np.uint8)


@registry.nodes.add("pow2")
class Square(NodeImageProcessing):
    def __init__(self):
        super().__init__("pow2", "POW", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return (cv2.pow((x[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)


@registry.nodes.add("exp")
class Exp(NodeImageProcessing):
    def __init__(self):
        super().__init__("exp", "EXP", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return (cv2.exp((x[0] / 255.0).astype(np.float32), 2) * 255).astype(np.uint8)


@registry.nodes.add("log")
class Log(NodeImageProcessing):
    def __init__(self):
        super().__init__("log", "LOG", 1, 0, sources="Numpy")

    def call(self, x, args=None):
        return np.log1p(x[0]).astype(np.uint8)


@registry.nodes.add("median_blur")
class MedianBlur(NodeImageProcessing):
    def __init__(self):
        super().__init__("median_blur", "BLRM", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        return cv2.medianBlur(x[0], ksize)


@registry.nodes.add("gaussian_blur")
class GaussianBlur(NodeImageProcessing):
    def __init__(self):
        super().__init__("gaussian_blur", "BLRG", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        return cv2.GaussianBlur(x[0], (ksize, ksize), 0)


@registry.nodes.add("laplacian")
class Laplacian(NodeImageProcessing):
    def __init__(self):
        super().__init__("laplacian", "LPLC", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.Laplacian(x[0], cv2.CV_64F).astype(np.uint8)


@registry.nodes.add("sobel")
class Sobel(NodeImageProcessing):
    def __init__(self):
        super().__init__("sobel", "SOBL", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        if args[1] < 128:
            return cv2.Sobel(x[0], cv2.CV_64F, 1, 0, ksize=ksize).astype(np.uint8)
        return cv2.Sobel(x[0], cv2.CV_64F, 0, 1, ksize=ksize).astype(np.uint8)


@registry.nodes.add("robert_cross")
class RobertCross(NodeImageProcessing):
    def __init__(self):
        super().__init__("robert_cross", "RBRT", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        img = (x[0] / 255.0).astype(np.float32)
        h = cv2.filter2D(img, -1, ROBERT_CROSS_H_KERNEL)
        v = cv2.filter2D(img, -1, ROBERT_CROSS_V_KERNEL)
        return (cv2.sqrt(cv2.pow(h, 2) + cv2.pow(v, 2)) * 255).astype(np.uint8)


@registry.nodes.add("canny")
class Canny(NodeImageProcessing):
    def __init__(self):
        super().__init__("canny", "CANY", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.Canny(x[0], args[0], args[1])


@registry.nodes.add("sharpen")
class Sharpen(NodeImageProcessing):
    def __init__(self):
        super().__init__("sharpen", "SHRP", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.filter2D(x[0], -1, SHARPEN_KERNEL)


@registry.nodes.add("gabor")
class GaborFilter(NodeImageProcessing):
    def __init__(self, ksize=11):
        super().__init__("gabor", "GABR", 1, 2, sources="OpenCV")
        self.ksize = ksize

    def call(self, x, args=None):
        gabor_k = gabor_kernel(self.ksize, args[0], args[1])
        return cv2.filter2D(x[0], -1, gabor_k)


@registry.nodes.add("abs_diff")
class AbsoluteDifference(NodeImageProcessing):
    """from https://github.com/cytosmart-bv/tomni"""

    def __init__(self):
        super().__init__("abs_diff", "ABSD", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        image = x[0].copy()
        return image - cv2.GaussianBlur(image, (ksize, ksize), 0) + args[1]


@registry.nodes.add("abs_diff2")
class AbsoluteDifference2(NodeImageProcessing):
    def __init__(self):
        super().__init__("abs_diff2", "ABS2", 2, 0, sources="OpenCV")

    def call(self, x, args=None):
        return 255 - cv2.absdiff(x[0], x[1])


@registry.nodes.add("fluo_tophat")
class FluoTopHat(NodeImageProcessing):
    """from https://github.com/cytosmart-bv/tomni"""

    def __init__(self):
        super().__init__("fluo_tophat", "FLUO", 1, 2, sources="Handmade")

    def _rescale_intensity(self, img, min_val, max_val):
        output_img = np.clip(img, min_val, max_val)
        if max_val - min_val == 0:
            return (output_img * 255).astype(np.uint8)
        output_img = (output_img - min_val) / (max_val - min_val) * 255
        return output_img.astype(np.uint8)

    def call(self, x, args=None):
        kernel = kernel_from_parameters(args)
        img = cv2.morphologyEx(x[0], cv2.MORPH_TOPHAT, kernel, iterations=10)
        kur = np.mean(kurtosis(img, fisher=True))
        skew1 = np.mean(skew(img))
        if kur > 1 and skew1 > 1:
            p2, p98 = np.percentile(img, (15, 99.5), interpolation="linear")
        else:
            p2, p98 = np.percentile(img, (15, 100), interpolation="linear")

        return self._rescale_intensity(img, p2, p98)


@registry.nodes.add("rel_diff")
class RelativeDifference(NodeImageProcessing):
    """from https://github.com/cytosmart-bv/tomni"""

    def __init__(self):
        super().__init__("rel_diff", "RELD", 1, 1, sources="Handmade")

    def call(self, x, args=None):
        img = x[0]
        max_img = np.max(img)
        min_img = np.min(img)

        ksize = correct_ksize(args[0])
        gb = cv2.GaussianBlur(img, (ksize, ksize), 0)
        gb = np.float32(gb)

        img = np.divide(img, gb + 1e-15, dtype=np.float32)
        img = cv2.normalize(img, img, max_img, min_img, cv2.NORM_MINMAX)
        return img.astype(np.uint8)


@registry.nodes.add("erode")
class Erode(ExportableNode):
    def __init__(self):
        super().__init__("erode", "EROD", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.erode(inputs[0], kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.erode({input_names[0]}, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::erode({input_names[0]}, {output_name}, kernel_from_parameters({args[0]}));"


@registry.nodes.add("dilate")
class Dilate(ExportableNode):
    def __init__(self):
        super().__init__("dilate", "DILT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.dilate(inputs[0], kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.dilate({input_names[0]}, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::dilate({input_names[0]}, {output_name}, kernel_from_parameters({args[0]}));"


@registry.nodes.add("open")
class Open(ExportableNode):
    def __init__(self):
        super().__init__("open", "OPEN", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_OPEN, kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.morphologyEx({input_names[0]}, cv2.MORPH_OPEN, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::morphologyEx({input_names[0]}, {output_name}, MORPH_OPEN, kernel_from_parameters({args[0]}));"


@registry.nodes.add("close")
class Close(ExportableNode):
    def __init__(self):
        super().__init__("close", "CLSE", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_CLOSE, kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.morphologyEx({input_names[0]}, cv2.MORPH_CLOSE, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::morphologyEx({input_names[0]}, {output_name}, MORPH_CLOSE, kernel_from_parameters({args[0]}));"


@registry.nodes.add("morph_gradient")
class MorphGradient(ExportableNode):
    def __init__(self):
        super().__init__("morph_gradient", "MGRD", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_GRADIENT, kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.morphologyEx({input_names[0]}, cv2.MORPH_GRADIENT, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::morphologyEx({input_names[0]}, {output_name}, MORPH_GRADIENT, kernel_from_parameters({args[0]}));"


@registry.nodes.add("morph_tophat")
class MorphTopHat(ExportableNode):
    def __init__(self):
        super().__init__("morph_tophat", "MTHT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_TOPHAT, kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.morphologyEx({input_names[0]}, cv2.MORPH_TOPHAT, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::morphologyEx({input_names[0]}, {output_name}, MORPH_TOPHAT, kernel_from_parameters({args[0]}));"


@registry.nodes.add("morph_blackhat")
class MorphBlackHat(ExportableNode):
    def __init__(self):
        super().__init__("morph_blackhat", "MBHT", 1, 2, sources="OpenCV")

    def call(self, inputs, p):
        kernel = kernel_from_parameters(p)
        return cv2.morphologyEx(inputs[0], cv2.MORPH_BLACKHAT, kernel)

    def to_python(self, input_names, p, output_name):
        return "{output_name} = cv2.morphologyEx({input_names[0]}, cv2.MORPH_BLACKHAT, kernel_from_parameters({args[0]}))"

    def to_cpp(self, input_names, p, output_name):
        return "cv::morphologyEx({input_names[0]}, {output_name}, MORPH_BLACKHAT, kernel_from_parameters({args[0]}));"


@registry.nodes.add("fill_holes")
class FillHoles(ExportableNode):
    def __init__(self):
        super().__init__("fill_holes", "FILL", 1, 0, sources="Handmade")

    def call(self, inputs, p):
        return morph_fill(inputs[0])

    def to_python(self, input_names, p, output_name):
        return "{output_name} = imfill({input_names[0]})"

    def to_cpp(self, input_names, p, output_name):
        return "imfill({input_names[0]}, {output_name});"


@registry.nodes.add("remove_small_objects")
class RemoveSmallObjects(NodeImageProcessing):
    def __init__(self):
        super().__init__("remove_small_objects", "RMSO", 1, 1, sources="Skimage")

    def call(self, x, args=None):
        return remove_small_objects(x[0] > 0, args[0]).astype(np.uint8)


@registry.nodes.add("remove_small_holes")
class RemoveSmallHoles(NodeImageProcessing):
    def __init__(self):
        super().__init__("remove_small_holes", "RMSH", 1, 1, sources="Skimage")

    def call(self, x, args=None):
        return remove_small_holes(x[0] > 0, args[0]).astype(np.uint8)


@registry.nodes.add("threshold")
class Threshold(NodeImageProcessing):
    def __init__(self):
        super().__init__("threshold", "TRH", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        if args[0] < 128:
            return threshold_binary(x[0], args[1])
        return threshold_tozero(x[0], args[1])


@registry.nodes.add("threshold_at_1")
class ThresholdAt1(NodeImageProcessing):
    def __init__(self):
        super().__init__("threshold_at_1", "TRH1", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        if args[0] < 128:
            return threshold_binary(x[0], 1)
        return threshold_tozero(x[0], 1)


# @registry.nodes.add("TRHA")
class ThresholdAdaptive(NodeImageProcessing):
    def __init__(self):
        super().__init__("adaptive_threshold", "TRHA", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        ksize = correct_ksize(args[0])
        C = args[1] - 128  # to allow negative values
        return cv2.adaptiveThreshold(
            x[0],
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ksize,
            C,
        )


@registry.nodes.add("distance_transform")
class DistanceTransform(NodeImageProcessing):
    def __init__(self):
        super().__init__("distance_transform", "DTRF", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.normalize(
            cv2.distanceTransform(x[0].copy(), cv2.DIST_L2, 3),
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            cv2.CV_8U,
        )


@registry.nodes.add("distance_transform_and_thresh")
class DistanceTransformAndThresh(NodeImageProcessing):
    def __init__(self):
        super().__init__(
            "distance_transform_and_thresh", "DTTR", 1, 2, sources="OpenCV"
        )

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


@registry.nodes.add("inrange_bin")
class BinaryInRange(NodeImageProcessing):
    def __init__(self):
        super().__init__("inrange_bin", "BRNG", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        lower = int(min(args[0], args[1]))
        upper = int(max(args[0], args[1]))
        return cv2.inRange(x[0], lower, upper)


@registry.nodes.add("inrange")
class InRange(NodeImageProcessing):
    def __init__(self):
        super().__init__("inrange", "RNG", 1, 2, sources="OpenCV")

    def call(self, x, args=None):
        lower = int(min(args[0], args[1]))
        upper = int(max(args[0], args[1]))
        return cv2.bitwise_and(
            x[0],
            x[0],
            mask=cv2.inRange(x[0], lower, upper),
        )


IMAGE_NODES_ABBV_LIST = registry.nodes.list().keys()
