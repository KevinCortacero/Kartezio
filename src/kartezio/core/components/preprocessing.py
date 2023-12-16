from abc import ABC

from kartezio.core.components.base import Node


class Preprocessing(Node, ABC):
    """
    Preprocessing node, called before training loop.
    """

    pass
