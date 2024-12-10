from abc import ABC
from typing import Dict

import numpy as np
from kartezio.callback import Event
from kartezio.components.core import UpdatableComponent, register


class MutationEffect(UpdatableComponent, ABC):
    def __init__(self):
        super().__init__()
        self.max_value = 256

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "MutationEffect":
        pass


@register(MutationEffect, "uniform")
class MutationUniform(MutationEffect):
    def adjust(self, _, new_parameters):
        return new_parameters

    def update(self, event: dict):
        pass

    def _precompute(self):
        return None


@register(MutationEffect, "weighted")
class MutationWeighted(MutationEffect):
    def __init__(self, start=0.5, end=None):
        super().__init__()
        self.start = start
        self.end = end
        self.weight = start

    def _precompute(self):
        if self.end is None:
            self.end = self.start
        return np.linspace(self.start, self.end, self.n_iterations)

    def adjust(self, old_parameters, new_parameters):
        return np.round(
            self.weight * new_parameters + (1 - self.weight) * old_parameters
        ).astype(np.uint8)

    def update(self, event: Event):
        if event.name == Event.Events.END_STEP:
            self.weight = self.stored[event.iteration]


@register(MutationEffect, "normal")
class MutationNormal(MutationEffect):
    def __init__(self, start=0.5, end=None):
        super().__init__()
        self.start = start
        self.end = end
        self.sigma = start * 255

    def _precompute(self):
        if self.end is None:
            self.end = self.start
        return np.linspace(self.start, self.end, self.n_iterations) * 255

    def adjust(self, old_parameters, new_parameters):
        return np.clip(np.random.normal(old_parameters, self.sigma), 0, 255)

    def update(self, event: Event):
        if event.name == Event.Events.END_STEP:
            self.sigma = self.stored[event.iteration]
