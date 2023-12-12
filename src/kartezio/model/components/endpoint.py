from abc import ABC
from typing import Dict, List

from kartezio.model.components.base import Components, Node
from kartezio.model.types import KType


class Endpoint(Node, ABC):
    """
    Last node called to produce final outputs. Called in training loop,
    not submitted to evolution.
    """

    def __init__(self, inputs: List[KType]):
        super().__init__()
        self.inputs = inputs

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Endpoint":
        return Components.instantiate(
            "Endpoint", dict_infos["name"], *dict_infos["args"]
        )
