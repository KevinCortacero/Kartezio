from math import sqrt
from typing import List

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

from kartezio.components.base import register
from kartezio.components.library import Library, Primitive
from kartezio.core.types import TypeArray, TypeScalar


@register(Library, "opencv_scalar")
class LibraryScalar(Library):
    def __init__(self):
        super().__init__(TypeScalar)


@register(Primitive, "const")
class Const(Primitive):
    def __init__(self):
        super().__init__([], TypeScalar, 1)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(args[0])


@register(Primitive, "max_value")
class MaxValue(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.max(x[0]))


@register(Primitive, "min_value")
class MinValue(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.min(x[0]))


@register(Primitive, "mean_value")
class MeanValue(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.mean(x[0]))


@register(Primitive, "median_value")
class MedianValue(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return float(np.median(x[0]))


@register(Primitive, "add_values")
class AddValues(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] + x[1], 255.0)


@register(Primitive, "subtract_values")
class SubtractValues(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(x[0] - x[1], 0)


@register(Primitive, "multiply_values")
class MultiplyValues(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] * x[1], 255.0)


@register(Primitive, "pow_value")
class PowValue(Primitive):
    def __init__(self):
        super().__init__([TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] ** 2, 255.0)


@register(Primitive, "sqrt_value")
class SqrtValue(Primitive):
    def __init__(self):
        super().__init__([TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return sqrt(x[0])


@register(Primitive, "multiply_by_2")
class MultiplyBy2(Primitive):
    def __init__(self):
        super().__init__([TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0] * 2.0, 255.0)


@register(Primitive, "divide_by_2")
class DivideBy2(Primitive):
    def __init__(self):
        super().__init__([TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return x[0] / 2.0


@register(Primitive, "min_scalars")
class MinScalars(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return min(x[0], x[1])


@register(Primitive, "max_scalars")
class MaxScalars(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(x[0], x[1])


@register(Primitive, "mean_scalars")
class MeanScalars(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return (x[0] + x[1]) / 2.0


@register(Primitive, "less_than")
class LessThan(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        if x[0] < x[1]:
            return 1.0
        return 0.0


@register(Primitive, "greater_than")
class GreaterThan(Primitive):
    def __init__(self):
        super().__init__([TypeScalar, TypeScalar], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        if x[0] > x[1]:
            return 1.0
        return 0.0


@register(Primitive, "skew")
class Skew(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(skew(x[0].reshape(-1)), 0.0)


@register(Primitive, "kurtosis")
class Kurtosis(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return max(kurtosis(x[0].reshape(-1)), 0.0)


@register(Primitive, "mean_abs_diff")
class MeanAbsDiff(Primitive):
    def __init__(self):
        super().__init__([TypeArray] * 2, TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return np.mean(cv2.absdiff(x[0], x[1]))


@register(Primitive, "coverage")
class Coverage(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return np.count_nonzero(x[0]) / (x[0].shape[0] * x[0].shape[0]) * 255.0


@register(Primitive, "count_regions")
class CountRegions(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeScalar, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return len(np.unique(cv2.connectedComponents(x[0], connectivity=4)[1]))


library_scalar = LibraryScalar()
library_scalar.add_by_name("const")
library_scalar.add_by_name("max_value")
library_scalar.add_by_name("min_value")
library_scalar.add_by_name("mean_value")
library_scalar.add_by_name("median_value")
library_scalar.add_by_name("add_values")
library_scalar.add_by_name("subtract_values")
library_scalar.add_by_name("multiply_values")
library_scalar.add_by_name("pow_value")
library_scalar.add_by_name("sqrt_value")
library_scalar.add_by_name("multiply_by_2")
library_scalar.add_by_name("divide_by_2")
library_scalar.add_by_name("min_scalars")
library_scalar.add_by_name("max_scalars")
library_scalar.add_by_name("mean_scalars")
library_scalar.add_by_name("less_than")
library_scalar.add_by_name("greater_than")
library_scalar.add_by_name("mean_abs_diff")
# library_scalar.add_by_name("coverage")
# library_scalar.add_by_name("count_regions")
# library_scalar.add_by_name("skew")
# library_scalar.add_by_name("kurtosis")
