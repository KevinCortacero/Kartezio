from abc import ABC
from typing import Dict, List

import numpy as np

from kartezio.core.components.base import Component, register
from kartezio.core.components.genotype import Genotype, MonoChromosome


class Adapter(Component, ABC):
    pass


@register(Adapter, "mono")
class AdapterMono(Adapter):
    """
    Adpater Design Pattern: https://refactoring.guru/design-patterns/adapter
    """

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Component":
        pass

    def new(self):
        return self.prototype.clone()

    def __init__(self, n_inputs, n_nodes, n_outputs, n_connections, n_parameters):
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
        self.prototype = self.create_prototype()

    def create_prototype(self):
        return MonoChromosome(
            self.n_outputs, np.zeros((self.n_nodes, self.w), dtype=np.uint8)
        )

    def write_function(self, genome, node, function_id):
        genome[node, 0] = function_id

    def write_connections(self, genome, node, connections):
        genome[node, self.con_idx : self.para_idx] = connections

    def write_parameters(self, genome, node, parameters):
        genome[node, self.para_idx :] = parameters

    def write_output_connection(self, genome, output_index, connection):
        genome.outputs[output_index] = connection

    def read_function(self, genome, node):
        return genome[node, 0]

    def read_connections(self, genome, node):
        return genome[node, self.con_idx : self.para_idx]

    def read_active_connections(self, genome, node, active_connections):
        return genome[
            node,
            self.con_idx : self.con_idx + active_connections,
        ]

    def read_parameters(self, genome, node):
        return genome[node, self.para_idx :]

    def read_outputs(self, genotype):
        return genotype.outputs
