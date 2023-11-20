import numpy as np

from kartezio.improc.primitives import LibraryDefaultOpenCV, library_opencv
from kartezio.model.base import ModelBase
from kartezio.model.components import (
    Library,
    Endpoint,
    DecoderSequential,
)
from kartezio.model.evolution import Fitness


def iou(y_true: np.ndarray, y_pred: np.ndarray):
        _y_true = y_true[0]
        _y_pred = y_pred["mask"]
        _y_pred[_y_pred > 0] = 1
        if np.sum(_y_true) == 0:
            _y_true = 1 - _y_true
            _y_pred = 1 - _y_pred
        intersection = np.logical_and(_y_true, _y_pred)
        union = np.logical_or(_y_true, _y_pred)
        return np.sum(intersection) / np.sum(union)

class FitnessIOU(Fitness):
    def __init__(self):
        super().__init__(iou)

class ModelSequential(ModelBase):
    def __init__(
        self, inputs: int, nodes: int, library: Library, fitness: FitnessIOU(), endpoint: Endpoint = None
    ):
        super().__init__(DecoderSequential(inputs, nodes, library, endpoint), fitness)


if __name__ == "__main__":
    model = ModelSequential(2, 30, library_opencv, FitnessIOU())
    model.compile(100, 4)
    model.fit()
