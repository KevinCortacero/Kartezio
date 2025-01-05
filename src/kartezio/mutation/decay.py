from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from kartezio.callback import Event
from kartezio.core.components import UpdatableComponent, fundamental, register
from kartezio.mutation.base import Mutation


@fundamental()
class MutationDecay(UpdatableComponent, ABC):
    def __init__(self):
        super().__init__()
        self._mutation = None

    def set_mutation(self, mutation: Mutation):
        self._mutation = mutation

    def update(self, event: Event):
        if event.name == Event.Events.END_STEP:
            self._mutation.node_rate = self.stored[event.iteration]


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

    def __init__(self, start: float, end: float):
        super().__init__()
        self.start = start
        self.end = end

    def _precompute(self):
        return np.linspace(self.start, self.end, self.n_iterations)


@register(MutationDecay)
class DegreeDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "DegreeDecay":
        pass

    def __init__(self, degree: int, start: float, end: float):
        super().__init__()
        self.degree = degree
        self.start = start
        self.end = end

    def _precompute(self):
        x = np.linspace(0, 1, self.n_iterations)
        return (self.end - self.start) * np.power(x, self.degree) + self.start


@register(MutationDecay)
class InvDegreeDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "InvDegreeDecay":
        pass

    def __init__(self, degree: int, start: float, end: float):
        super().__init__()
        self.degree = degree
        self.start = start
        self.end = end

    def _precompute(self):
        x = np.linspace(1, 0, self.n_iterations)
        return -(self.end - self.start) * np.power(x, self.degree) + self.end
