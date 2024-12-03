from abc import ABC
from typing import Dict, List

from kartezio.components.core import Components, Node
from kartezio.types import KType


class Endpoint(Node, ABC):
    """
    Represents the final node in a CGP graph, responsible for producing the final outputs.

    The Endpoint is invoked in the training loop but is not involved in the evolutionary process.
    """

    def __init__(self, inputs: List[KType]):
        """
        Initialize an Endpoint instance.

        Args:
            inputs (List[KType]): The list of inputs for the endpoint node.
        """
        super().__init__()
        self.inputs = inputs

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Endpoint":
        """
        Create an Endpoint instance from a dictionary representation.

        Args:
            dict_infos (Dict): A dictionary containing the name and arguments for the Endpoint.

        Returns:
            Endpoint: A new Endpoint instance created from the given dictionary.
        """
        return Components.instantiate(
            "Endpoint",
            dict_infos["name"].lower().replace(" ", "_"),
            **dict_infos["args"]
        )
