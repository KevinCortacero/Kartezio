from abc import ABC
from typing import Dict

from kartezio.core.components.base import Node, Components


class Preprocessing(Node, ABC):
    """
    Preprocessing node, called before training loop.
    """

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Preprocessing":
        return Components.instantiate(
            "Preprocessing", dict_infos["name"], *dict_infos["args"]
        )
