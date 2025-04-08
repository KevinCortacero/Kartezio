from math import sqrt
from typing import List

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

from kartezio.core.components import Library, Primitive, register
from kartezio.types import Matrix, Scalar


@register(Primitive)
class Const(Primitive):
    def __init__(self):
        super().__init__([], Scalar, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(args[0])


@register(Primitive)
class MaxValue(Primitive):
    def __init__(self):
        super().__init__([Matrix], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.max(x[0]))


@register(Primitive)
class MinValue(Primitive):
    def __init__(self):
        super().__init__([Matrix], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.min(x[0]))


@register(Primitive)
class MeanValue(Primitive):
    def __init__(self):
        super().__init__([Matrix], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.mean(x[0]))


@register(Primitive)
class MedianValue(Primitive):
    def __init__(self):
        super().__init__([Matrix], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.median(x[0]))


@register(Primitive)
class AddValues(Primitive):
    def __init__(self):
        super().__init__([Scalar, Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] + x[1], 255.0)


@register(Primitive)
class SubtractValues(Primitive):
    def __init__(self):
        super().__init__([Scalar, Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(x[0] - x[1], 0)


@register(Primitive)
class MultiplyValues(Primitive):
    def __init__(self):
        super().__init__([Scalar, Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] * x[1], 255.0)


@register(Primitive)
class PowValue(Primitive):
    def __init__(self):
        super().__init__([Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] ** 2, 255.0)


@register(Primitive)
class SqrtValue(Primitive):
    def __init__(self):
        super().__init__([Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return sqrt(x[0])


@register(Primitive)
class MultiplyBy2(Primitive):
    def __init__(self):
        super().__init__([Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] * 2.0, 255.0)


@register(Primitive)
class DivideBy2(Primitive):
    def __init__(self):
        super().__init__([Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return x[0] / 2.0


@register(Primitive)
class MinScalars(Primitive):
    def __init__(self):
        super().__init__([Scalar, Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0], x[1])


@register(Primitive)
class MaxScalars(Primitive):
    def __init__(self):
        super().__init__([Scalar, Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(x[0], x[1])


@register(Primitive)
class MeanScalars(Primitive):
    def __init__(self):
        super().__init__([Scalar, Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return (x[0] + x[1]) / 2.0


@register(Primitive)
class LessThan(Primitive):
    def __init__(self):
        super().__init__([Scalar, Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        if x[0] < x[1]:
            return 1.0
        return 0.0


@register(Primitive)
class GreaterThan(Primitive):
    def __init__(self):
        super().__init__([Scalar, Scalar], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        if x[0] > x[1]:
            return 1.0
        return 0.0


@register(Primitive)
class Skew(Primitive):
    def __init__(self):
        super().__init__([Matrix], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(skew(x[0].reshape(-1)), 0.0)


@register(Primitive)
class Kurtosis(Primitive):
    def __init__(self):
        super().__init__([Matrix], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(kurtosis(x[0].reshape(-1)), 0.0)


@register(Primitive)
class MeanAbsDiff(Primitive):
    def __init__(self):
        super().__init__([Matrix] * 2, Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return np.mean(cv2.absdiff(x[0], x[1]))


@register(Primitive)
class Coverage(Primitive):
    def __init__(self):
        super().__init__([Matrix], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return np.count_nonzero(x[0]) / (x[0].shape[0] * x[0].shape[0]) * 255.0


@register(Primitive)
class CountRegions(Primitive):
    def __init__(self):
        super().__init__([Matrix], Scalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return len(np.unique(cv2.connectedComponents(x[0], connectivity=4)[1]))


library_scalar = Library(Scalar)
library_scalar.add_primitive(Const())
library_scalar.add_primitive(MaxValue())
library_scalar.add_primitive(MinValue())
library_scalar.add_primitive(MeanValue())
library_scalar.add_primitive(MedianValue())
library_scalar.add_primitive(AddValues())
library_scalar.add_primitive(SubtractValues())
library_scalar.add_primitive(MultiplyValues())
library_scalar.add_primitive(PowValue())
library_scalar.add_primitive(SqrtValue())
library_scalar.add_primitive(MultiplyBy2())
library_scalar.add_primitive(DivideBy2())
library_scalar.add_primitive(MinScalars())
library_scalar.add_primitive(MaxScalars())
library_scalar.add_primitive(MeanScalars())
library_scalar.add_primitive(LessThan())
library_scalar.add_primitive(GreaterThan())
library_scalar.add_primitive(Skew())
library_scalar.add_primitive(Kurtosis())
library_scalar.add_primitive(MeanAbsDiff())
library_scalar.add_primitive(Coverage())
library_scalar.add_primitive(CountRegions())
