import numpy as np

from kartezio.callback import CallbackVerbose
from kartezio.dataset import read_dataset
from kartezio.improc.primitives import library_opencv
from kartezio.model.evolution import Fitness
from kartezio.model.sequential import ModelSequential


def iou(y_true: np.ndarray, y_pred: np.ndarray):
    _y_true = y_true[0]
    _y_pred = y_pred[0]
    _y_pred[_y_pred > 0] = 1
    if np.sum(_y_true) == 0:
        _y_true = 1 - _y_true
        _y_pred = 1 - _y_pred
    intersection = np.logical_and(_y_true, _y_pred)
    union = np.logical_or(_y_true, _y_pred)
    return 1 - np.sum(intersection) / np.sum(union)


class FitnessIOU(Fitness):
    def __init__(self, reduction="mean", multiprocessing=False):
        super().__init__(iou, reduction, multiprocessing)


if __name__ == "__main__":
    path = r"/home/kevin.cortacero/Repositories/KartezioPaper/cell_image_library/dataset"
    dataset = read_dataset(path, indices=[0, 1, 2, 3])
    draft = ModelSequential(3, 30, library_opencv, FitnessIOU(reduction="mean"))

    model = draft.compile(200, 4, callbacks=[CallbackVerbose()])
    elite, history = model.fit(dataset.train_x, dataset.train_y)
    p, t = model.predict(dataset.train_x)
