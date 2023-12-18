import ast
import copy
from abc import ABC
from typing import Dict

import numpy as np

from kartezio.core.components.base import Component, register


class Genotype(Component, ABC):
    """
    Only store "DNA" into a Numpy array (ndarray)
    No metadata stored in DNA to avoid duplicates
    Avoiding RAM overload: https://refactoring.guru/design-patterns/flyweight
    Default genotype would be: 3 inputs, 10 function nodes and 1 output (M=14),
    with 1 function, 2 connections and 2 parameters (N=5), giving final 2D shape (14, 5).
    """

    def clone(self):
        return copy.deepcopy(self)


@register(Genotype, "mono_chromosome")
class MonoChromosome(Genotype):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "MonoChromosome":
        pass

    def __init__(self, n_outputs, chromosome: np.ndarray):
        super().__init__()
        self.chromosome = chromosome
        self.outputs = np.zeros(n_outputs, dtype=np.uint8)

    def __copy__(self):
        new = self.__class__(len(self.outputs), self.chromosome.copy())
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        new = self.__class__(len(self.outputs), self.chromosome.copy())
        new.chromosome = self.chromosome.copy()
        new.outputs = self.outputs.copy()
        return new

    def __getitem__(self, item):
        return self.chromosome.__getitem__(item)

    def __setitem__(self, key, value):
        return self.chromosome.__setitem__(key, value)

    @staticmethod
    def from_ndarray(chromosome: np.ndarray, outputs: np.ndarray):
        genotype = MonoChromosome(len(outputs), chromosome)
        genotype.outputs = outputs
        return genotype

    @classmethod
    def from_json(cls, json_data):
        genes = np.asarray(ast.literal_eval(json_data["sequence"]))
        return cls.from_ndarray(genes)
