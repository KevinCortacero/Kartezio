from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from kartezio.model.components.base import Components, Node


class Primitive(Node, ABC):
    """
    Primitive function called inside the CGP Graph.
    """

    def __init__(self, input_types: List, rtype, n_parameters: int):
        super().__init__()
        self.input_types = input_types
        self.rtype = rtype
        self.arity = len(input_types)
        self.n_parameters = n_parameters

    @abstractmethod
    def call(self, x: List[np.ndarray], args: List[int]):
        pass

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Primitive":
        return Components.instantiate("Primitive", dict_infos["name"])
