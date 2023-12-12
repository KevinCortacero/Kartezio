from abc import ABC

from kartezio.model.components.base import Node


class Preprocessing(Node, ABC):
    """
    Preprocessing node, called before training loop.
    """

    pass
