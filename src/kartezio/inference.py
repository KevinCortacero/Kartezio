from abc import ABC, abstractmethod

import numpy as np
from kartezio.components.core import Components
from kartezio.components.genotype import Genotype
from kartezio.evolution.decoder import Decoder
from kartezio.plot import plot_mask
from kartezio.utils.directory import Directory
from kartezio.utils.io import JsonLoader
from kartezio.vision.common import bgr2rgb


class InferenceModel(ABC):
    @abstractmethod
    def predict(self, x, reformat_x=None):
        pass


class CodeModel(InferenceModel, ABC):
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def call_node(self, node_name, x, args):
        return Components.instantiate("Primitive", node_name).call(x, args)

    @abstractmethod
    def _parse(self, X):
        pass

    def _endpoint(self, x):
        if self.endpoint is None:
            return x
        return self.endpoint.call(x)

    def predict(self, x, preprocessing=None):
        if preprocessing:
            x = preprocessing.call(x)
        return [self._endpoint(self._parse(xi)) for xi in x]


class EnsembleModel(InferenceModel):
    def __init__(self, models):
        self.models = models

    def predict(self, x):
        return [model.predict(x) for model in self.models]

    def remove_endpoint(self):
        for model in self.models:
            model.endpoint = None


class ModelPool:
    def __init__(self, directory, fitness, regex=""):
        self.models = []
        if type(directory) == str:
            self.directory = Directory(directory)
        else:
            self.directory = directory
        self.fitness = fitness
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
        self.decoder = decoder
        self.preprocessing = preprocessing
        self.fitness = fitness
        self.indices = dataset["indices"]

    def preprocess(self, x):
        if self.preprocessing:
            return self.preprocessing.call(x)
        return x
    
    def predict(self, x):
        """
        Predict the output of the model given the input.
        Apply the preprocessing if it exists.
        """
        x = self.preprocess(x)
        return self.decoder.decode(self.genotype, x)
    
    def evaluate(self, x, y):
        y_pred, _ = self.predict(x)
        return self.fitness.batch(y, [y_pred])

    """
    def show_graph(self, inputs, outputs, jupyter=False):
        return show_graph(self._model, inputs, outputs, jupyter=jupyter)
    """

    def plot_predictions(self, dataset, subset="test"):
        x, y, v = dataset.train_xyv if subset == "train" else dataset.test_xyv
        p, t = self.predict(x)
        for visual_image, y_true, y_pred in zip(v, y, p):
            plot_mask(bgr2rgb(visual_image), y_pred["mask"], gt=y_true[0])
