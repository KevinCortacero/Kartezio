from math import sqrt
from typing import List

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

from kartezio.core.components import Library, Primitive, register
from kartezio.types import TypeArray, TypeScalar


@register(Primitive)
class Const(Primitive):
    def __init__(self):
        super().__init__([], TypeScalar, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(args[0])


@register(Primitive)
class MaxValue(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.max(x[0]))


@register(Primitive)
class MinValue(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.min(x[0]))


@register(Primitive)
class MeanValue(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.mean(x[0]))


@register(Primitive)
class MedianValue(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.median(x[0]))


@register(Primitive)
class AddValues(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] + x[1], 255.0)


@register(Primitive)
class SubtractValues(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(x[0] - x[1], 0)


@register(Primitive)
class MultiplyValues(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] * x[1], 255.0)


@register(Primitive)
class PowValue(Primitive):
    def __init__(self):
        super().__init__([TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] ** 2, 255.0)


@register(Primitive)
class SqrtValue(Primitive):
    def __init__(self):
        super().__init__([TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return sqrt(x[0])


@register(Primitive)
class MultiplyBy2(Primitive):
    def __init__(self):
        super().__init__([TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] * 2.0, 255.0)


@register(Primitive)
class DivideBy2(Primitive):
    def __init__(self):
        super().__init__([TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return x[0] / 2.0


@register(Primitive)
class MinScalars(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0], x[1])


@register(Primitive)
class MaxScalars(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(x[0], x[1])


@register(Primitive)
class MeanScalars(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return (x[0] + x[1]) / 2.0


@register(Primitive)
class LessThan(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        if x[0] < x[1]:
            return 1.0
        return 0.0


@register(Primitive)
class GreaterThan(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        if x[0] > x[1]:
            return 1.0
        return 0.0


@register(Primitive)
class Skew(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(skew(x[0].reshape(-1)), 0.0)


@register(Primitive)
class Kurtosis(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(kurtosis(x[0].reshape(-1)), 0.0)


@register(Primitive)
class MeanAbsDiff(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return np.mean(cv2.absdiff(x[0], x[1]))


@register(Primitive)
class Coverage(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return np.count_nonzero(x[0]) / (x[0].shape[0] * x[0].shape[0]) * 255.0


@register(Primitive)
class CountRegions(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return len(np.unique(cv2.connectedComponents(x[0], connectivity=4)[1]))


library_scalar = Library(TypeScalar)
library_scalar.add_primitive(Const())
library_scalar.add_primitive("max_value")
library_scalar.add_primitive("min_value")
library_scalar.add_primitive("mean_value")
library_scalar.add_primitive("median_value")
library_scalar.add_primitive("add_values")
library_scalar.add_primitive("subtract_values")
library_scalar.add_primitive("multiply_values")
library_scalar.add_primitive("pow_value")
library_scalar.add_primitive("sqrt_value")
library_scalar.add_primitive("multiply_by_2")
library_scalar.add_primitive("divide_by_2")
library_scalar.add_primitive("min_scalars")
library_scalar.add_primitive("max_scalars")
library_scalar.add_primitive("mean_scalars")
library_scalar.add_primitive("less_than")
library_scalar.add_primitive("greater_than")
library_scalar.add_primitive("mean_abs_diff")
# library_scalar.add_primitive("coverage")
# library_scalar.add_primitive("count_regions")
# library_scalar.add_primitive("skew")
# library_scalar.add_primitive("kurtosis")
