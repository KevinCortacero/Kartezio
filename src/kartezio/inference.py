from abc import ABC, abstractmethod

import cv2
import numpy as np

from kartezio.core.components import Components, Endpoint, Fitness
from kartezio.evolution.decoder import Decoder
from kartezio.types import DataBatch, DataPopulation, Parameters
from kartezio.utils.directory import Directory
from kartezio.utils.io import JsonLoader


class InferenceModel(ABC):
    @abstractmethod
    def predict(self, x: DataBatch, preprocessing=None) -> DataBatch:
        pass


class CodeModel(InferenceModel, ABC):
    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint

    def call_node(self, node_name: str, x: DataBatch, args: Parameters):
        return Components.instantiate("Primitive", node_name).call(x, args)

    @abstractmethod
    def _parse(self, x: DataBatch) -> DataBatch:
        pass

    def _endpoint(self, x: DataBatch):
        if self.endpoint is None:
            return x
        return self.endpoint.call(x)

    def predict(self, x: DataBatch, preprocessing=None) -> DataBatch:
        if preprocessing:
            x = preprocessing.call(x)
        return [self._endpoint(self._parse(xi)) for xi in x]


class EnsembleModel(InferenceModel):
    def __init__(self, models: list[InferenceModel]):
        self.models = models
        self.normalize = True
        self.reduction = "mean"
        self.erosion = None

    def batch(self, x: DataBatch) -> DataPopulation:
        return [model.predict(x) for model in self.models]

    def predict(self, x: DataBatch, preprocessing=None) -> DataBatch:
        y_batch = self.batch(x)
        y_list = []
        for i in range(len(x)):
            mask_list = []
            for pi in y_batch:
                one_image = pi[0][i][0]
                if self.normalize:
                    one_image = cv2.normalize(
                        one_image, None, 0.0, 1.0, cv2.NORM_MINMAX
                    )
                else:
                    one_image = one_image / 255.0
                mask_list.append(one_image)
            y = np.array(mask_list)
            if self.reduction == "mean":
                y = y.mean(axis=0)
            elif self.reduction == "max":
                y = y.max(axis=0)
            elif self.reduction == "min":
                y = y.min(axis=0)
            y = (y * 255).astype(np.uint8)
            if self.erosion:
                y = cv2.erode(y, np.ones((self.erosion, self.erosion)), iterations=1)
            y_list.append(y)
        return y_list

    def remove_endpoint(self):
        for model in self.models:
            model.endpoint = None


class ModelPool:
    def __init__(self, directory, regex=""):
        self.models = []
        if isinstance(directory, str):
            self.directory = Directory(directory)
        else:
            self.directory = directory
        self.read(regex)

    def read(self, regex):
        for model in self.directory.ls(f"{regex}", ordered=True):
            self.add_model(model)

    def add_model(self, filepath):
        model = KartezioModel(filepath)
        self.models.append(model)

    def sample_ensemble_model(self, n):
        indices = np.random.randint(0, len(self.models), n)
        models = [self.models[i] for i in indices]
        return EnsembleModel(models)

    def to_ensemble(self):
        return EnsembleModel(self.models)


class KartezioModel(InferenceModel):
    """Wrapper"""

    json_loader = JsonLoader()

    def __init__(self, filepath: str):
        super().__init__()
        dataset, genotype, decoder, preprocessing, fitness = (
            KartezioModel.json_loader.read_individual(filepath=filepath)
        )
        self.genotype = genotype
        self.decoder: Decoder = decoder
        self.preprocessing = preprocessing
        self.fitness: Fitness = fitness
        self.indices = dataset["indices"]

    def preprocess(self, x):
        if self.preprocessing:
            return self.preprocessing.call(x)
        return x

    def predict(self, x: DataBatch) -> DataBatch:
        """
        Predict the output of the model given the input.
        Apply the preprocessing if it exists.
        """
        x = self.preprocess(x)
        return self.decoder.decode(self.genotype, x)

    def evaluate(self, x: DataBatch, y_true: DataBatch):
        y_pred, _ = self.predict(x)
        return self.fitness.batch(y_true, [y_pred])
