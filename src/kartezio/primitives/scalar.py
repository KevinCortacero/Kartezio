from math import sqrt

import cv2
import numpy as np
from scipy.stats import kurtosis, skew

from kartezio.core.components import Library, Primitive, register
from kartezio.types import (
    ArrayData,
    DataList,
    DataType,
    Matrix1,
    Matrix2,
    Parameters,
    Scalar1,
    Scalar2,
)


@register(Primitive)
class Const(Primitive):
    def __init__(self):
        super().__init__([], DataType.SCALAR, 1)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return args[0]


@register(Primitive)
class MaxValue(Primitive):
    def __init__(self):
        super().__init__(Matrix1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return np.max(x[0])


@register(Primitive)
class MinValue(Primitive):
    def __init__(self):
        super().__init__(Matrix1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return np.min(x[0])


@register(Primitive)
class MeanValue(Primitive):
    def __init__(self):
        super().__init__(Matrix1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return np.mean(x[0])


@register(Primitive)
class MedianValue(Primitive):
    def __init__(self):
        super().__init__(Matrix1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return np.median(x[0])


@register(Primitive)
class AddValues(Primitive):
    def __init__(self):
        super().__init__(Scalar2, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return min(x[0] + x[1], 255)


@register(Primitive)
class SubtractValues(Primitive):
    def __init__(self):
        super().__init__(Scalar2, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return max(x[0] - x[1], 0)


@register(Primitive)
class MultiplyValues(Primitive):
    def __init__(self):
        super().__init__(Scalar2, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return min(x[0] * x[1], 255)


@register(Primitive)
class PowValue(Primitive):
    def __init__(self):
        super().__init__(Scalar1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return min(x[0] ** 2, 255)


@register(Primitive)
class SqrtValue(Primitive):
    def __init__(self):
        super().__init__(Scalar1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return sqrt(x[0])


@register(Primitive)
class MultiplyBy2(Primitive):
    def __init__(self):
        super().__init__(Scalar1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return min(x[0] * 2, 255)


@register(Primitive)
class DivideBy2(Primitive):
    def __init__(self):
        super().__init__(Scalar1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return x[0] // 2


@register(Primitive)
class MinScalars(Primitive):
    def __init__(self):
        super().__init__(Scalar2, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return min(x[0], x[1])


@register(Primitive)
class MaxScalars(Primitive):
    def __init__(self):
        super().__init__(Scalar2, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return max(x[0], x[1])


@register(Primitive)
class MeanScalars(Primitive):
    def __init__(self):
        super().__init__(Scalar2, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return (x[0] + x[1]) // 2


@register(Primitive)
class LessThan(Primitive):
    def __init__(self):
        super().__init__(Scalar2, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        if x[0] < x[1]:
            return 1
        return 0


@register(Primitive)
class GreaterThan(Primitive):
    def __init__(self):
        super().__init__(Scalar2, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        if x[0] > x[1]:
            return 1
        return 0


@register(Primitive)
class Skew(Primitive):
    def __init__(self):
        super().__init__(Matrix1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return min(max(skew(x[0].reshape(-1)), 0), 255)


@register(Primitive)
class Kurtosis(Primitive):
    def __init__(self):
        super().__init__(Matrix1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return min(max(kurtosis(x[0].reshape(-1)), 0), 255)


@register(Primitive)
class MeanAbsDiff(Primitive):
    def __init__(self):
        super().__init__(Matrix2, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return np.mean(cv2.absdiff(x[0], x[1]))


@register(Primitive)
class Coverage(Primitive):
    def __init__(self):
        super().__init__(Matrix1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return np.count_nonzero(x[0]) / (x[0].shape[0] * x[0].shape[1]) * 255


@register(Primitive)
class CountRegions(Primitive):
    def __init__(self):
        super().__init__(Matrix1, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters) -> ArrayData:
        return min(
            len(np.unique(cv2.connectedComponents(x[0], connectivity=4)[1])),
            255,
        )


def default_scalar_lib(include_matrix=False) -> Library:
    """
    Create a default library of scalar operations.

    Returns:
        Library: A library containing various scalar operations.
    """
    library_scalar = Library(DataType.SCALAR)
    library_scalar.add_primitive(Const())
    if include_matrix:
        library_scalar.add_primitive(MaxValue())
        library_scalar.add_primitive(MinValue())
        library_scalar.add_primitive(MeanValue())
        library_scalar.add_primitive(MedianValue())
        # library_scalar.add_primitive(Skew()) # TODO: fix and adapt for images
        # library_scalar.add_primitive(Kurtosis()) # TODO: fix and adapt for images
        library_scalar.add_primitive(MeanAbsDiff())
        library_scalar.add_primitive(Coverage())
        library_scalar.add_primitive(CountRegions())
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
    return library_scalar
