from abc import ABC, abstractmethod
import random
from tabulate import tabulate

from kartezio.model.primitive import KPrimitive
from kartezio.model.registry import registry
from kartezio.model.components import KartezioComponent


class KLibrary(KartezioComponent, ABC):
    def __init__(self):
        self._primitives = {}
        self.fill()

    @staticmethod
    def from_json(json_data):
        library = EmptyLibrary()
        for node_name in json_data:
            library.add_primitive(node_name)
        return library

    @abstractmethod
    def fill(self):
        pass

    def add_primitive(self, primitive: KPrimitive):
        self._primitives[len(self._primitives)] = primitive

    def add_library(self, library):
        for f in library.primitives:
            self.add_primitive(f.name)

    def name_of(self, i):
        return self._primitives[i].name

    def symbol_of(self, i):
        return self._primitives[i].symbol

    def arity_of(self, i):
        return self._primitives[i].arity

    def parameters_of(self, i):
        return self._primitives[i].p

    def execute(self, name, x, args):
        return self._primitives[name].call(x, args)

    def display(self):
        headers = ["Id", "Name", "Symbol", "Inputs", "Outputs", "Parameters"]
        full_list = []
        for i, primitive in self._primitives.items():
            print(i, primitive)
            one_primitive_infos = [
                i,
                primitive.signature.name,
                primitive.symbol,
                primitive.signature.input_types,
                primitive.signature.output_type,
                primitive.signature.n_parameters,
            ]
            full_list.append(one_primitive_infos)
        print(tabulate(full_list, headers=headers))

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

    @property
    def ordered_list(self):
        return [self._primitives[i].name for i in range(self.size)]

    def dumps(self) -> dict:
        return {}


class EmptyLibrary(KLibrary):
    def fill(self):
        pass