from enum import StrEnum
from typing import Sequence, TypeAlias

import numpy as np

# --- Scores ---
Score: TypeAlias = np.ndarray
ScoreBatch: TypeAlias = np.ndarray
ScorePopulation: TypeAlias = np.ndarray


# --- Runtime values ---
ArrayData: TypeAlias = np.ndarray


# --- Logical data kind ---
class DataType(StrEnum):
    SCALAR = "scalar"  # shape: (1,)
    VECTOR = "vector"  # shape: (l, 1)
    MATRIX = "matrix"  # shape: (h, w)
    TENSOR = "tensor"  # shape: (c, h, w)


Signature: TypeAlias = Sequence[DataType]


def Matrix(n: int) -> Signature:
    return [DataType.MATRIX] * n


def Scalar(n: int) -> Signature:
    return [DataType.SCALAR] * n


Matrix1: Signature = Matrix(1)
Matrix2: Signature = Matrix(2)
Scalar1: Signature = Scalar(1)
Scalar2: Signature = Scalar(2)
MatrixScalar: Signature = [DataType.MATRIX, DataType.SCALAR]


# --- Containers ---
DataList: TypeAlias = list[ArrayData]
DataBatch: TypeAlias = list[DataList]
DataPopulation: TypeAlias = list[DataBatch]

Parameters: TypeAlias = Sequence[int]
