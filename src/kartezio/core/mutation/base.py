from abc import ABC, abstractmethod

import numpy as np

from kartezio.core.components.base import Component
from kartezio.core.components.genotype import Genotype


class Mutation(Component, ABC):
    def __init__(self, decoder):
        super(Component).__init__()
        self.decoder = decoder
        self.n_functions = decoder.library.size
        self.parameter_max_value = 256

    @property
    def random_parameters(self):
        return np.random.randint(
            self.parameter_max_value, size=self.decoder.adapter.n_parameters
        )

    @property
    def random_functions(self):
        return np.random.randint(self.n_functions)

    @property
    def random_output(self):
        return np.random.randint(self.decoder.adapter.out_idx, size=1)

    def random_connections(self, idx: int):
        return np.random.randint(
            self.decoder.adapter.n_inputs + idx,
            size=self.decoder.adapter.n_connections,
        )

    def mutate_function(self, genome: Genotype, idx: int):
        self.decoder.adapter.write_function(genome, idx, self.random_functions)

    def mutate_connections(self, genome: Genotype, idx: int, only_one: int = None):
        new_connections = self.random_connections(idx)
        if only_one is not None:
            new_value = new_connections[only_one]
            new_connections = self.decoder.adapter.read_connections(genome, idx)
            new_connections[only_one] = new_value
        self.decoder.adapter.write_connections(genome, idx, new_connections)

    def mutate_parameters(self, genome: Genotype, idx: int, only_one: int = None):
        new_parameters = self.random_parameters
        if only_one is not None:
            old_parameters = self.decoder.adapter.read_parameters(genome, idx)
            old_parameters[only_one] = new_parameters[only_one]
            new_parameters = old_parameters.copy()
        self.decoder.adapter.write_parameters(genome, idx, new_parameters)

    def mutate_output(self, genome: Genotype, idx: int):
        self.decoder.adapter.write_output_connection(genome, idx, self.random_output)

    @abstractmethod
    def mutate(self, genome: Genotype):
        pass
