from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np

from kartezio.types import DataBatch, DataList
from kartezio.utils.directory import Directory
from kartezio.vision.common import draw_overlay

CSV_DATASET = "dataset.csv"


class Dataset:
    class SubSet:
        def __init__(self):
            self.x: DataBatch = []
            self.y: DataBatch = []
            self.v: DataBatch = []

        def add_item(self, x: DataBatch, y: DataBatch):
            self.x.append(x)
            self.y.append(y)

        def add_visual(self, visual):
            self.v.append(visual)

        @property
        def xy(self):
            return self.x, self.y

        @property
        def xyv(self):
            return self.x, self.y, self.v

    def __init__(self, training: SubSet, test: SubSet, inputs, indices=None):
        self.training = training
        self.test = test
        self.inputs = inputs
        self.indices = indices

    @property
    def train_x(self):
        return self.training.x

    @property
    def train_y(self):
        return self.training.y

    @property
    def train_v(self):
        return self.training.v

    @property
    def test_x(self):
        return self.test.x

    @property
    def test_y(self):
        return self.test.y

    @property
    def test_v(self):
        return self.test.v

    @property
    def train_xy(self):
        return self.training.xy

    @property
    def test_xy(self):
        return self.test.xy

    @property
    def train_xyv(self):
        return self.training.xyv

    @property
    def test_xyv(self):
        return self.test.xyv

    @property
    def split(self):
        return self.train_x, self.train_y, self.test_x, self.test_y


class DataReader:
    def __init__(self, directory, scale=1.0):
        self.scale = scale
        self.directory = directory

    def read(self, filename, shape=None):
        if str(filename) == "nan":
            filepath = ""
        else:
            filepath = f"{self.directory}/{filename}"
        return self._read(filepath, shape)

    @abstractmethod
    def _read(self, filepath, shape=None):
        pass


@dataclass
class DatasetReader(Directory):
    x_reader: DataReader
    y_reader: DataReader
    preview: bool = False
    preview_dir: Directory = field(init=False)
    color_preview = (51, 152, 75)

    def __post_init__(self, path):
        super().__post_init__(path)
        if self.preview:
            self.preview_dir = self.next("__preview__")

    def read_dataset(
        self,
        dataset_filename=CSV_DATASET,
        indices=None,
    ):
        return self._read_from_dataframe(dataset_filename, indices)

    def _read_from_dataframe(self, dataset_filename, indices):
        dataframe = self.read(dataset_filename)
        dataframe_training = dataframe[dataframe["set"] == "training"]
        training = self._read_dataset(dataframe_training, indices)
        dataframe_testing = dataframe[dataframe["set"] == "test"]
        testing = self._read_dataset(dataframe_testing)
        input_sizes = []
        [input_sizes.append(len(xi)) for xi in training.x]
        [input_sizes.append(len(xi)) for xi in testing.x]
        input_sizes = np.array(input_sizes)
        inputs = int(input_sizes[0])
        if not np.all((input_sizes == inputs)):
            """
            raise ValueError(
                f"Inconsistent size of inputs for this dataset: sizes: {input_sizes}"
            )
            """
            print(f"Inconsistent size of inputs for this dataset: sizes: {input_sizes}")

        if self.preview:
            for i in range(len(training.x)):
                visual = training.v[i]
                label = training.y[i][0]
                preview = draw_overlay(
                    visual,
                    label.astype(np.uint8),
                    color=self.color_preview,
                    alpha=0.5,
                    thickness=3,
                )
                self.preview_dir.write(f"train_{i}.png", preview)
            for i in range(len(testing.x)):
                visual = testing.v[i]
                label = testing.y[i][0]
                preview = draw_overlay(
                    visual,
                    label.astype(np.uint8),
                    color=self.color_preview,
                    alpha=0.5,
                    thickness=3,
                )
                self.preview_dir.write(f"test_{i}.png", preview)
        return Dataset(training, testing, inputs, indices)

    def _read_auto(self, dataset):
        pass

    def _read_dataset(self, dataframe, indices=None):
        dataset = Dataset.SubSet()
        dataframe.reset_index(inplace=True)
        if indices:
            dataframe = dataframe.loc[indices]
        for row in dataframe.itertuples():
            x = self.x_reader.read(row.input, shape=None)
            y = self.y_reader.read(row.label, shape=x.shape)
            dataset.n_inputs = x.size
            dataset.add_item(x.datalist, y.datalist)
            visual_from_table = False
            if "visual" in dataframe.columns:
                if str(row.visual) != "nan":
                    dataset.add_visual(self.read(row.visual))
                    visual_from_table = True
            if not visual_from_table:
                dataset.add_visual(x.visual)
        return dataset


@dataclass
class DataItem:
    datalist: DataList
    shape: tuple
    count: int
    visual: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.datalist)


def read_dataset(
    dataset_path,
    x_reader: DataReader | str,
    y_reader: DataReader | str,
    filename=CSV_DATASET,
    indices=None,
    preview=False,
):
    from kartezio.readers import (
        ImageGrayscaleReader,
        ImageLabels,
        ImageRGBReader,
        RoiPolygonReader,
        RoiPolyhedronReader,
        TiffImageGray3dReader,
        TiffImageLabel3dReader,
    )

    if isinstance(x_reader, str):
        if x_reader == "rgb":
            x_reader = ImageRGBReader(dataset_path)
        elif x_reader == "grayscale":
            x_reader = ImageGrayscaleReader(dataset_path)
        elif x_reader == "tiff_mono_3d":
            x_reader = TiffImageGray3dReader(dataset_path)
        elif x_reader == "tiff_labels_3d":
            x_reader = TiffImageLabel3dReader(dataset_path)
        else:
            raise ValueError(f"unnknown x_reader: {x_reader}")
    if isinstance(y_reader, str):
        if y_reader == "labels":
            y_reader = ImageLabels(dataset_path)
        elif y_reader == "imagej":
            y_reader = RoiPolygonReader(dataset_path)
        elif y_reader == "tiff_labels_3d":
            y_reader = TiffImageLabel3dReader(dataset_path)
        elif y_reader == "polyhedron":
            y_reader = RoiPolyhedronReader(dataset_path)
        else:
            raise ValueError(f"unnknown y_reader: {y_reader}")

    dataset_reader = DatasetReader(dataset_path, x_reader, y_reader, preview=preview)
    return dataset_reader.read_dataset(dataset_filename=filename, indices=indices)
