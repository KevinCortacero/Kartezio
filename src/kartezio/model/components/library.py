from typing import Dict

from kartezio.model.components.base import Component
from kartezio.model.components.primitive import Primitive


class Library(Component):
    def __init__(self, return_type):
        self._primitives: Dict[int, Primitive] = {}
        self.return_type = return_type

    def to_toml(self):
        return {
            "return_type": self.return_type,
            "primitives": {
                str(i): self.name_of(i) for i in range(len(self._primitives))
            },
        }

    @staticmethod
    def from_json(json_data):
        library = Library()
        for node_name in json_data:
            library.add_primitive(node_name)
        return library

    def add_primitive(self, primitive: Primitive):
        self._primitives[len(self._primitives)] = primitive

    def add_library(self, library):
        for p in library.primitives:
            self.add_primitive(p)

    def name_of(self, i):
        return self._primitives[i].name

    def arity_of(self, i):
        return len(self._primitives[i].inputs)

    def parameters_of(self, i):
        return self._primitives[i].parameters

    def execute(self, f_index, x, args):
        return self._primitives[f_index](x, args)

    def display(self):
        headers = ["Id", "Name", "Inputs", "Outputs", "Arity", "Parameters"]
        full_list = []
        for i, primitive in self._primitives.items():
            one_primitive_infos = [
                i,
                self.name_of(i),
                primitive.inputs,
                primitive.output,
                self.arity_of(i),
                self.parameters_of(i),
            ]
            full_list.append(one_primitive_infos)
        table_name = f"  {self.output_type} Library  "
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
