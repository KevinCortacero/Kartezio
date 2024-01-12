from abc import ABC
from typing import Dict

from kartezio.callback import Event
from kartezio.core.components.base import UpdatableComponent, register
from kartezio.core.mutation.base import Mutation


class MutationDecay(UpdatableComponent, ABC):
    def __init__(self):
        super().__init__()
        self._mutation = None

    def set_mutation(self, mutation: Mutation):
        self._mutation = mutation


@register(MutationDecay, "linear")
class LinearDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "LinearDecay":
        pass

    def __init__(self, step: float):
        super().__init__()
        self.step = step

    def update(self, event: dict):
        if event["name"] == Event.END_STEP:
            self._mutation.node_rate -= self.step
            self._mutation.out_rate -= self.step


@register(MutationDecay, "factor")
class FactorDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "FactorDecay":
        pass

    def __init__(self, decay_factor: float):
        super().__init__()
        self.decay_factor = decay_factor

    def update(self, event: dict):
        if event["name"] == Event.END_STEP:
            self._mutation.node_rate *= self.decay_factor
            self._mutation.out_rate *= self.decay_factor
