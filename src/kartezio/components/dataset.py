from abc import abstractmethod
from ast import List
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from kartezio.components.core import Component
from kartezio.utils.json_handler import json_read, json_write


class DataReader(Component):
    def __init__(self, directory, scale=1.0):
        super().__init__()
        self.scale = scale
        self.directory = directory

    def read(self, filename, shape=None):
        if str(filename) == "nan":
            filepath = ""
        else:
            filepath = str(self.directory / filename)
        return self._read(filepath, shape)

    @abstractmethod
    def _read(self, filepath, shape=None):
        pass

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Component":
        pass


class DatasetMeta:
    @staticmethod
    def write(
        filepath,
        name,
        input_type,
        input_format,
        label_type,
        label_format,
        label_name,
        scale=1.0,
        mode="dataframe",
        meta_filename="META.json",
    ):
        json_data = {
            "name": name,
            "scale": scale,
            "label_name": label_name,
            "mode": mode,
            "input": {"type": input_type, "format": input_format},
            "label": {"type": label_type, "format": label_format},
        }
        json_write(filepath + "/" + meta_filename, json_data)

    @staticmethod
    def read(filepath, meta_filename):
        return json_read(filepath / meta_filename)


@dataclass
class DataItem:
    datalist: List
    shape: Tuple
    count: int
    visual: np.ndarray = None

    @property
    def size(self):
        return len(self.datalist)
