import random
from abc import ABC
from typing import Dict, List

import numpy as np
from kartezio.components.core import Component, Components, Node, register
from kartezio.types import KType
from tabulate import tabulate


class Primitive(Node, ABC):
    """
    Primitive function called inside the CGP Graph.
    """

    def __init__(self, inputs: List[KType], output: KType, n_parameters: int):
        super().__init__()
        self.inputs = inputs
        self.output = output
        self.arity = len(inputs)
        self.n_parameters = n_parameters

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Primitive":
        return Components.instantiate("Primitive", dict_infos["name"])
    
    def __to_dict__(self) -> Dict:
        return {"name": self.name}


class Library(Component, ABC):
    def __init__(self, rtype):
        super().__init__()
        self._primitives: Dict[int, Primitive] = {}
        self.rtype = rtype

    def __to_dict__(self) -> Dict:
        return {
            "rtype": self.rtype,
            "primitives": {i: self.name_of(i) for i in range(self.size)},
        }

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Library":
        rtype = dict_infos["rtype"]
        library = LibraryEmpty(rtype)
        size = len(dict_infos["primitives"])
        for i in range(size):
            library.add_by_name(dict_infos["primitives"][str(i)])
        return library

    def add_by_name(self, name):
        primitive = Components.instantiate("Primitive", name)
        self.add_primitive(primitive)

    def add_primitive(self, primitive: Primitive):
        self._primitives[len(self._primitives)] = primitive

    def add_library(self, library):
        for p in library.primitives:
            self.add_primitive(p)

    def name_of(self, i):
        return self._primitives[i].name

    def arity_of(self, i):
        return self._primitives[i].arity

    def parameters_of(self, i):
        return self._primitives[i].n_parameters

    def inputs_of(self, i):
        return self._primitives[i].inputs

    def execute(self, f_index, x: List[np.ndarray], args: List[int]):
        return self._primitives[f_index].call(x, args)

    def display(self):
        headers = ["Id", "Name", "Inputs", "Outputs", "Arity", "Parameters"]
        full_list = []
        for i, primitive in self._primitives.items():
            one_primitive_infos = [
                i,
                self.name_of(i),
                self.inputs_of(i),
                primitive.rtype,
                self.arity_of(i),
                self.parameters_of(i),
            ]
            full_list.append(one_primitive_infos)
        table_name = f"  {self.rtype} Library  "
        print("─" * len(table_name))
        print(table_name)
        print(
            tabulate(
                full_list,
                tablefmt="simple_grid",
                headers=headers,
                numalign="center",
                stralign="center",
            )
        )

    @property
    def random_index(self):
        return random.choice(self.keys)

    @property
    def last_index(self):
        return len(self._primitives) - 1

    @property
    def primitives(self):
        return list(self._primitives.values())

    @property
    def keys(self):
        return list(self._primitives.keys())

    @property
    def max_arity(self):
        return max([self.arity_of(i) for i in self.keys])

    @property
    def max_parameters(self):
        return max([self.parameters_of(i) for i in self.keys])

    @property
    def size(self):
        return len(self._primitives)


@register(Library, "library")
class LibraryEmpty(Library):
    pass
