import numpy as np

from kartezio.callback import CallbackVerbose
from kartezio.dataset import read_dataset
from kartezio.improc.primitives import library_opencv
from kartezio.model.evolution import Fitness
from kartezio.model.sequential import ModelSequential


if __name__ == "__main__":
    path = r"/home/kevin.cortacero/Repositories/KartezioPaper/cell_image_library/dataset"
    dataset = read_dataset(path, indices=[0, 1, 2, 3])
    draft = ModelSequential(3, 30, library_opencv, FitnessIOU(reduction="mean"))

    model = draft.compile(200, 4, callbacks=[CallbackVerbose()])
    elite, history = model.fit(dataset.train_x, dataset.train_y)
    p, t = model.predict(dataset.train_x)
