from abc import ABC, abstractmethod
from typing import List

from kartezio.core.components.base import Node


class Reduction(Node, ABC):
    def call(self, x: List):
        y = []
        for xi in x:
            y.append(self.reduce(xi))
        return y

    @abstractmethod
    def reduce(self, x):
        pass
