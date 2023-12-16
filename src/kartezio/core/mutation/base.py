from abc import ABC, abstractmethod

import numpy as np

from kartezio.core.components.base import Component
from kartezio.core.components.genotype import Genotype
from kartezio.core.components_old import GenomeReaderWriter


class Mutation(Component, GenomeReaderWriter, ABC):
    def __init__(self, shape, n_functions):
        super().__init__(shape)
        self.n_functions = n_functions
        self.parameter_max_value = 256

    def dumps(self) -> dict:
        return {}

    @property
    def random_parameters(self):
        return np.random.randint(self.parameter_max_value, size=self.infos.n_parameters)

    @property
    def random_functions(self):
        return np.random.randint(self.n_functions)

    @property
    def random_output(self):
        return np.random.randint(self.infos.out_idx, size=1)

    def random_connections(self, idx: int):
        return np.random.randint(
            self.infos.nodes_idx + idx, size=self.infos.n_connections
        )

    def mutate_function(self, genome: Genotype, idx: int):
        self.write_function(genome, idx, self.random_functions)

    def mutate_connections(self, genome: Genotype, idx: int, only_one: int = None):
        new_connections = self.random_connections(idx)
        if only_one is not None:
            new_value = new_connections[only_one]
            new_connections = self.read_connections(genome, idx)
            new_connections[only_one] = new_value
        self.write_connections(genome, idx, new_connections)

    def mutate_parameters(self, genome: Genotype, idx: int, only_one: int = None):
        new_parameters = self.random_parameters
        if only_one is not None:
            old_parameters = self.read_parameters(genome, idx)
            old_parameters[only_one] = new_parameters[only_one]
            new_parameters = old_parameters.copy()
        self.write_parameters(genome, idx, new_parameters)

    def mutate_output(self, genome: Genotype, idx: int):
        self.write_output_connection(genome, idx, self.random_output)

    @abstractmethod
    def mutate(self, genome: Genotype):
        pass
