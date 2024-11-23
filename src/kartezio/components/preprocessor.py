from abc import ABC, abstractmethod
from typing import Dict

from kartezio.components.base import Components, Node


class Preprocessing(Node, ABC):
    """
    Preprocessing node, called before training loop.
    """

    def __init__(self):
        super().__init__()
        self.__then = None

    def call(self, x, args=None):
        if self.__then is not None:
            return self.__then.call(self.preprocess(x), args)
        return self.preprocess(x)

    @abstractmethod
    def preprocess(self, x):
        raise NotImplementedError

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Preprocessing":
        return Components.instantiate(
            "Preprocessing", dict_infos["name"], **dict_infos["args"]
        )

    def then(self, preprocessing: "Preprocessing"):
        if self.__then is not None:
            self.__then.then(preprocessing)
        else:
            self.__then = preprocessing
        return self
