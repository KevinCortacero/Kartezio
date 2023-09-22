from abc import ABC
import random
from tabulate import tabulate

from kartezio.model.primitive import KPrimitive
from kartezio.model.components import KartezioComponent


class KLibrary(KartezioComponent, ABC):
    def __init__(self, output_type):
        self._primitives = {}
        self.output_type = output_type

    @staticmethod
    def from_json(json_data):
        library = EmptyLibrary()
        for node_name in json_data:
            library.add_primitive(node_name)
        return library

    def add_primitive(self, primitive: KPrimitive):
        self._primitives[len(self._primitives)] = primitive

    def add_library(self, library):
        for p in library.primitives:
            self.add_primitive(p)

    def name_of(self, i):
        return self._primitives[i].signature.name

    def symbol_of(self, i):
        return self._primitives[i].symbol

    def arity_of(self, i):
        return self._primitives[i].signature.arity

    def parameters_of(self, i):
        return self._primitives[i].signature.n_parameters

    def execute(self, name, x, args):
        return self._primitives[name](x, args)

    def display(self):
        headers = ["Id", "Name", "Symbol", "Inputs", "Outputs", "Arity", "Param."]
        full_list = []
        for i, primitive in self._primitives.items():
            one_primitive_infos = [
                i, self.name_of(i), self.symbol_of(i),
                primitive.signature.input_types,
                primitive.signature.output_type,
                self.arity_of(i),
                self.parameters_of(i),
            ]
            full_list.append(one_primitive_infos)
        table_name = f"  {self.output_type} Library  "
        print("─" * len(table_name))
        print(table_name)
        print(tabulate(full_list, tablefmt="simple_grid", headers=headers, numalign="center", stralign="center"))

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
        return [self.name_of(i) for i in range(self.size)]

    def dumps(self) -> dict:
        return {}


class EmptyLibrary(KLibrary):
    def fill(self):
        pass