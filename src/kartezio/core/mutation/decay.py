from abc import ABC
from typing import Dict

from kartezio.callback import Event
from kartezio.core.components.base import UpdatableComponent, register
from kartezio.core.mutation.base import Mutation


class MutationDecay(UpdatableComponent, ABC):
    def __init__(self):
        super().__init__()
        self.__mutation = None

    def set_mutation(self, mutation: Mutation):
        self.__mutation = mutation


class NoDecay(MutationDecay):
    def __init__(self, mutation: Mutation):
        super().__init__("no_decay", mutation)

    def update(self, event):
        pass


class LinearDecay(MutationDecay):
    def __init__(
        self, mutation: Mutation, start: float, end: float, max_iterations: int
    ):
        super().__init__("linear_decay", mutation)
        self.start = start
        self.end = end
        self.max_iterations = max_iterations
        self.__step = (self.start - self.end) / self.max_iterations

    def update(self, event: dict):
        if event["name"] == Event.END_STEP:
            n = event["n"]
            new_node_rate = self.start - self.__step * n
            print(n, new_node_rate)
            self._mutation.node_rate = new_node_rate


@register(MutationDecay, "factor")
class FactorDecay(MutationDecay):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Component":
        pass

    def __init__(self, decay_factor: float):
        super().__init__()
        self.decay_factor = decay_factor

    def update(self, event: dict):
        if event["name"] == Event.END_STEP:
            n = event["n"]
            self._mutation.node_rate *= self.decay_factor
            print(n, self._mutation.node_rate)
