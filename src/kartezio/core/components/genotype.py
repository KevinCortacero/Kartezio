import ast
import copy
from typing import Dict

import numpy as np

from kartezio.core.components.base import Component


class Genotype(Component):
    """
    Only store "DNA" into Numpy arrays (ndarray)
    No metadata stored in DNA to avoid duplicates
    https://refactoring.guru/design-patterns/flyweight
    """

    def __init__(self, n_outputs: int):
        super().__init__()
        self._chromosomes: Dict[str, np.ndarray] = {}
        self._chromosomes["outputs"] = np.zeros(n_outputs, dtype=np.uint8)

    def __getitem__(self, item: str):
        return self._chromosomes.__getitem__(item)

    def __setitem__(self, key: str, value: np.ndarray):
        self._chromosomes.__setitem__(key, value)

    def __deepcopy__(self, memo):
        new = self.__class__(len(self._chromosomes["outputs"]))
        for key, value in self._chromosomes.items():
            new._chromosomes[key] = value.copy()
        return new

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Genotype":
        assert "chromosomes" in dict_infos
        assert "outputs" in dict_infos["chromosomes"]
        n_outputs = len(dict_infos["chromosomes"]["outputs"])
        genotype = Genotype(n_outputs)
        for key, value in dict_infos["chromosomes"].items():
            genotype[key] = np.asarray(value)
        return genotype

    def clone(self):
        return copy.deepcopy(self)
