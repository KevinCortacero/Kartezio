from abc import ABC
from typing import Dict

import numpy as np

from kartezio.callback import Event
from kartezio.core.components.base import UpdatableComponent, register


class MutationEffect(UpdatableComponent, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "MutationEffect":
        pass


@register(MutationEffect, "uniform")
class MutationUniform(MutationEffect):
    def call(self, _, new_parameters):
        return new_parameters

    def update(self, event: dict):
        pass


@register(MutationEffect, "weighted")
class MutationWeighted(MutationEffect):
    def __init__(self, step, weight=0.5):
        super().__init__()
        self.weight = weight
        self.step = step

    def call(self, old_parameters, new_parameters):
        return (
            self.weight * new_parameters + (1 - self.weight) * old_parameters
        )

    def update(self, event: dict):
        if event["name"] == Event.END_STEP:
            self.weight -= self.step


@register(MutationEffect, "normal")
class MutationNormal(MutationEffect):
    def __init__(self, step, sigma=128):
        super().__init__()
        self.sigma = sigma
        self.step = step

    def call(self, old_parameters, new_parameters):
        return np.clip(np.random.normal(old_parameters, self.sigma), 0, 255)

    def update(self, event: dict):
        if event["name"] == Event.END_STEP:
            self.sigma -= self.step
