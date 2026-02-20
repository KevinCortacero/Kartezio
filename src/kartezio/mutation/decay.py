from abc import ABC
from typing import Dict

import numpy as np

from kartezio.callback import Event
from kartezio.core.components import UpdatableComponent, fundamental, register
from kartezio.mutation.base import Mutation


@fundamental()
class MutationDecay(UpdatableComponent, ABC):
    def __init__(self, discrete_bins: int = None):
        super().__init__()
        self._mutation = None
        self.discrete_bins = discrete_bins

    def set_mutation(self, mutation: Mutation):
        self._mutation = mutation

    def update(self, event: Event):
        if event.name == Event.Events.END_STEP:
            self._mutation.node_rate = self.stored[event.iteration]

    def _discretize(self, values):
        if self.discrete_bins:
            bin_indices = np.linspace(0, len(values) - 1, self.discrete_bins, dtype=int)
            bin_size = len(values) // self.discrete_bins
            for i, index in enumerate(bin_indices):
                values[i * bin_size : (i + 1) * bin_size] = values[index]
        return values


@register(MutationDecay)
class ConstantDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "ConstantDecay":
        pass

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def _precompute(self):
        return np.ones(self.n_iterations) * self.value


@register(MutationDecay)
class LinearDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "LinearDecay":
        pass

    def __init__(self, start: float, end: float, discrete_bins: int = None):
        super().__init__(discrete_bins)
        self.start = start
        self.end = end

    def _precompute(self):
        values = np.linspace(self.start, self.end, self.n_iterations)
        return self._discretize(values)


@register(MutationDecay)
class DegreeDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "DegreeDecay":
        pass

    def __init__(
        self, degree: int, start: float, end: float, discrete_bins: int = None
    ):
        super().__init__(discrete_bins)
        self.degree = degree
        self.start = start
        self.end = end

    def _precompute(self):
        x = np.linspace(0, 1, self.n_iterations)
        return self._discretize(
            (self.end - self.start) * np.power(x, self.degree) + self.start
        )


@register(MutationDecay)
class InvDegreeDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "InvDegreeDecay":
        pass

    def __init__(
        self, degree: int, start: float, end: float, discrete_bins: int = None
    ):
        super().__init__(discrete_bins)
        self.degree = degree
        self.start = start
        self.end = end

    def _precompute(self):
        x = np.linspace(1, 0, self.n_iterations)
        return self._discretize(
            -(self.end - self.start) * np.power(x, self.degree) + self.end
        )
