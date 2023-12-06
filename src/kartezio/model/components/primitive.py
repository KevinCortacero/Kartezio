from abc import ABC, abstractmethod
from typing import List

import numpy as np

from kartezio.model.components.base import Component


class Primitive(Component, ABC):
    """
    Primitive function called inside the CGP Graph.
    """

    def __init__(self, input_types: List, return_type, n_parameters: int):
        self.input_types = input_types
        self.return_type = return_type
        self.n_connections = len(input_types)
        self.n_parameters = n_parameters

    @abstractmethod
    def call(self, x: List[np.ndarray], args: List[int]):
        pass
