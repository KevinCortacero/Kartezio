from abc import ABC
from typing import Dict, List

import numpy as np

from kartezio.core.components.base import Component, register
from kartezio.core.components.genotype import MonoChromosome, MultiChromosomes


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

    def __init__(
        self, n_inputs, n_nodes, n_outputs, n_connections, n_parameters
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


@register(Adapter, "poly")
class AdapterPoly(Adapter):
    """
    Adpater Design Pattern: https://refactoring.guru/design-patterns/adapter
    """

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "AdapterPoly":
        pass

    def new(self):
        return self.prototype.clone()

    def __init__(
        self, n_inputs, n_nodes, returns, n_connections, n_parameters, rtypes
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.returns = returns
        self.n_outputs = len(self.returns)
        self.n_connections = n_connections
        self.n_parameters = n_parameters
        assert len(n_parameters) == len(n_parameters), "not uniform"
        self.types_map = {t: i for i, t in enumerate(rtypes)}
        assert (
            len(n_parameters) == len(n_parameters) == len(self.types_map)
        ), f"Libraries provided seem to have same return types: {rtypes}."
        self.in_idx = 0
        self.con_idx = 1
        self.out_idx = self.n_inputs + self.n_nodes
        self.para_idx = [self.con_idx + c for c in self.n_connections]
        self.w = [
            1 + self.n_connections[i] + self.n_parameters[i]
            for i in range(len(self.n_connections))
        ]
        self.prototype = self.create_prototype()

    def create_prototype(self):
        chromosomes = [
            np.zeros((self.n_nodes, wi), dtype=np.uint8) for wi in self.w
        ]
        return MultiChromosomes(self.n_outputs, chromosomes)

    def write_function(
        self,
        genotype: MultiChromosomes,
        chromosome: int,
        node: int,
        function_id,
    ):
        genotype.get_chromosome(chromosome)[node, 0] = function_id

    def write_connections(
        self,
        genotype: MultiChromosomes,
        chromosome: int,
        node: int,
        connections,
    ):
        genotype.get_chromosome(chromosome)[
            node, self.con_idx : self.para_idx[chromosome]
        ] = connections

    def write_parameters(
        self,
        genotype: MultiChromosomes,
        chromosome: int,
        node: int,
        parameters,
    ):
        genotype.get_chromosome(chromosome)[
            node, self.para_idx[chromosome] :
        ] = parameters

    def write_output_connection(
        self, genotype: MultiChromosomes, output_index, connection
    ):
        genotype.outputs[output_index] = connection

    def read_function(
        self, genotype: MultiChromosomes, chromosome: int, node: int
    ):
        return genotype.get_chromosome(chromosome)[node, 0]

    def read_connections(
        self, genotype: MultiChromosomes, chromosome: int, node: int
    ):
        return genotype.get_chromosome(chromosome)[
            node, self.con_idx : self.para_idx[chromosome]
        ]

    def read_active_connections(
        self,
        genotype: MultiChromosomes,
        chromosome: int,
        node: int,
        n_connections: int,
    ):
        return genotype.get_chromosome(chromosome)[
            node,
            self.con_idx : self.con_idx + n_connections,
        ]

    def read_parameters(
        self, genotype: MultiChromosomes, chromosome: int, node: int
    ):
        return genotype.get_chromosome(chromosome)[
            node, self.para_idx[chromosome] :
        ]

    def read_outputs(self, genotype: MultiChromosomes):
        return genotype.outputs

    def to_chromosome_indices(self, types):
        return [self.types_map[_type] for _type in types]
