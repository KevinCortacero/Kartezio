from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from kartezio.callback import Event
from kartezio.components.core import UpdatableComponent, register
from kartezio.mutation.base import Mutation


class MutationDecay(UpdatableComponent, ABC):
    def __init__(self):
        super().__init__()
        self._mutation = None

    def set_mutation(self, mutation: Mutation):
        self._mutation = mutation

    def update(self, event: Event):
        if event.name == Event.Events.END_STEP:
            self._mutation.node_rate = self.stored[event.iteration]


@register(MutationDecay, "linear")
class LinearDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "LinearDecay":
        pass

    def __init__(self, start: float, end: float):
        super().__init__()
        self.start = start
        self.end = end

    def _precompute(self):
        return np.linspace(self.start, self.end, self.n_iterations)


@register(MutationDecay, "geometric")
class GeometricDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "GeometricDecay":
        pass

    def __init__(self, start: float, end: float):
        super().__init__()
        self.start = start
        self.end = end

    def _precompute(self):
        return np.geomspace(self.start, self.end, self.n_iterations)


@register(MutationDecay, "geometric_inv")
class GeometricInvDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "GeometricInvDecay":
        pass

    def __init__(self, start: float, end: float):
        super().__init__()
        self.start = start
        self.end = end

    def _precompute(self):
        abs_diff = np.abs(
            np.geomspace(self.start, self.end, self.n_iterations)
            - np.linspace(self.start, self.end, self.n_iterations)
        )
        return np.linspace(self.start, self.end, self.n_iterations) + abs_diff


if __name__ == "__main__":
    n_iterations = 20000
    start = 0.15
    end = 0.01
    test_decay_linear = LinearDecay(start, end)
    test_decay_linear.compile(n_iterations)
    print(test_decay_linear.stored)

    test_decay_factor = GeometricInvDecay(start, end)
    test_decay_factor.compile(n_iterations)
    print(test_decay_factor.stored)

    test_decay_geometric = GeometricDecay(start, end)
    test_decay_geometric.compile(n_iterations)
    print(test_decay_geometric.stored)

    # plot 3 curves to compare the decays
    import matplotlib.pyplot as plt

    plt.plot(test_decay_linear.stored, label="Linear")
    plt.plot(test_decay_factor.stored, label="Factor")
    plt.plot(test_decay_geometric.stored, label="Geometric")
    plt.legend()
    plt.show()
