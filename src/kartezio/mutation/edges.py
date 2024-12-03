from abc import ABC
from typing import Dict

import numpy as np
from kartezio.components.core import Component, register


class MutationEdges(Component, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "MutationEdges":
        pass

    def random_output(self):
        pass

    def random_edges(self, idx: int, chromosome: int):
        pass


@register(MutationEdges, "uniform")
class MutationEdgesUniform(MutationEdges):
    def __init__(self):
        super().__init__()

    def weights_edges(self, idx: int):
        # return uniform distribution
        p = [1.0 / idx] * idx
        return p


@register(MutationEdges, "normal")
class MutationEdgesNormal(MutationEdges):
    def __init__(self, sigma=3):
        super().__init__()
        self.sigma = sigma

    def weights_edges(self, idx: int):
        # return half normal distribution
        p = np.exp(-np.power(np.arange(0, 1 + idx * 2) - idx, 2) / (2 * self.sigma**2))
        p = p[: 1 + idx]
        return p / np.sum(p)
