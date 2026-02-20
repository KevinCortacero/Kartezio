from abc import ABC

import numpy as np

from kartezio.core.components import KartezioComponent, fundamental, register


@fundamental()
class MutationEdges(KartezioComponent, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def __from_dict__(cls, dict_infos: dict) -> "MutationEdges":
        pass

    def random_output(self):
        pass

    def random_edges(self, idx: int, chromosome: int):
        pass


@register(MutationEdges)
class MutationEdgesUniform(MutationEdges):
    def __init__(self):
        super().__init__()

    def weights_edges(self, n: int):
        # return uniform distribution
        p = [1.0 / n] * n
        return p


@register(MutationEdges)
class MutationEdgesNormal(MutationEdges):
    def __init__(self, sigma=3):
        super().__init__()
        self.sigma = sigma
        self.w = {}

    def weights_edges(self, n: int):
        if n not in self.w:
            x = np.arange(-n + 1, 1)
            p = (
                1.0
                / (np.sqrt(2.0 * np.pi) * self.sigma)
                * np.exp(-np.power(x / self.sigma, 2.0) / 2)
            )
            self.w[n] = p / np.sum(p)
        return self.w[n]
