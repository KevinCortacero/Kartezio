from abc import ABC
from typing import Dict

import numpy as np

from kartezio.core.components.base import Component, register
from kartezio.core.components.genotype import Genotype


class Adapter(Component, ABC):
    """
    Adpater Design Pattern: https://refactoring.guru/design-patterns/adapter

    """

    def __init__(
        self,
        n_inputs: int,
        n_nodes: int,
        n_outputs: int,
        n_connections: int,
        n_parameters: int,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.n_outputs = n_outputs
        self.n_connections = n_connections
        self.n_parameters = n_parameters
        self.in_idx = 0
        self.con_idx = 1
        self.out_idx = self.n_inputs + self.n_nodes
        self.para_idx = self.con_idx + self.n_connections
        self.w = 1 + self.n_connections + self.n_parameters
        self._genotype = self._create_prototype()

    def _create_prototype(self) -> Genotype:
        raise NotImplementedError

    def new_genotype(self):
        return self._genotype.clone()


@register(Adapter, "mono")
class AdapterMono(Adapter):
    """
    Adpater Design Pattern: https://refactoring.guru/design-patterns/adapter
    """

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Component":
        pass

    def __init__(
        self, n_inputs, n_nodes, n_outputs, n_connections, n_parameters
    ):
        super().__init__(
            n_inputs, n_nodes, n_outputs, n_connections, n_parameters
        )

    def _create_prototype(self):
        return MonoChromosome(
            self.n_outputs, np.zeros((self.n_nodes, self.w), dtype=np.uint8)
        )

    def set_function(self, genotype: Genotype, node: int, function_id: int):
        genotype[node, 0] = function_id

    def set_connections(self, genotype: Genotype, node: int, connections):
        genotype[node, self.con_idx : self.para_idx] = connections

    def set_parameters(self, genotype: Genotype, node: int, parameters):
        genotype[node, self.para_idx :] = parameters

    def set_output_connection(
        self, genotype: Genotype, output_index: int, connection
    ):
        genotype.outputs[output_index] = connection

    def get_function(self, genotype, node):
        return genotype[node, 0]

    def get_connections(self, genotype, node):
        return genotype[node, self.con_idx : self.para_idx]

    def get_active_connections(self, genotype, node, active_connections):
        return genotype[
            node,
            self.con_idx : self.con_idx + active_connections,
        ]

    def get_parameters(self, genotype, node):
        return genotype[node, self.para_idx :]

    def get_outputs(self, genotype):
        return genotype.outputs
