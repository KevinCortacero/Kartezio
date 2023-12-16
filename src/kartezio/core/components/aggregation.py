from abc import ABC
from typing import List

from kartezio.core.components.base import Node


class Aggregation(Node, ABC):
    def __init__(
        self,
        aggregation,
        inputs,
        post_aggregation=None,
    ):
        super().__init__(aggregation)
        self.post_aggregation = post_aggregation
        self.inputs = inputs

    def call(self, x: List):
        y = []
        for i in range(len(self.inputs)):
            if self.post_aggregation:
                y.append(self.post_aggregation(self._fn(x[i])))
            else:
                y.append(self._fn(x[i]))
        return y
