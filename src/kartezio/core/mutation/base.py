from abc import ABC, abstractmethod

import numpy as np

from kartezio.core.components.base import Component
from kartezio.core.components.decoder import Decoder, DecoderPoly
from kartezio.core.components.genotype import Genotype
from kartezio.core.mutation.effect import MutationUniform


class Mutation(Component, ABC):
    def __init__(self, decoder: Decoder):
        super(Component).__init__()
        self.decoder = decoder
        self.parameter_max_value = 256
        self.effect = MutationUniform()

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

    def mutate_connections(
        self, genome: Genotype, idx: int, only_one: int = None
    ):
        new_connections = self.random_connections(idx)
        if only_one is not None:
            new_value = new_connections[only_one]
            new_connections = self.decoder.adapter.read_connections(
                genome, idx
            )
            new_connections[only_one] = new_value
        self.decoder.adapter.write_connections(genome, idx, new_connections)

    def mutate_parameters(
        self, genome: Genotype, idx: int, only_one: int = None
    ):
        old_parameters = self.decoder.adapter.read_parameters(genome, idx)
        new_random_parameters = self.random_parameters

        new_parameters = self.effect.call(
            old_parameters, new_random_parameters
        )
        if only_one is not None:
            old_parameters[only_one] = new_parameters[only_one]
            new_parameters = old_parameters.copy()
        self.decoder.adapter.write_parameters(genome, idx, new_parameters)

    def mutate_output(self, genome: Genotype, idx: int):
        self.decoder.adapter.write_output_connection(
            genome, idx, self.random_output
        )

    @abstractmethod
    def mutate(self, genome: Genotype):
        pass


class MutationPoly(Component, ABC):
    def __init__(self, decoder: DecoderPoly):
        super().__init__()
        self.decoder = decoder
        self.parameter_max_value = 256
        self.effect = MutationUniform()

    def random_parameters(self, chromosome: int):
        return np.random.randint(
            self.parameter_max_value,
            size=self.decoder.adapter.chromosomes_infos[
                chromosome
            ].n_parameters,
        )

    def random_function(self, chromosome: str):
        return np.random.randint(
            self.decoder.adapter.chromosomes_infos[chromosome].n_functions
        )

    @property
    def random_output(self):
        return np.random.randint(self.decoder.adapter.out_idx, size=1)

    def random_edges(self, idx: int, chromosome: int):
        return np.random.randint(
            self.decoder.adapter.n_inputs + idx,
            size=self.decoder.adapter.chromosomes_infos[chromosome].n_edges,
        )

    def mutate_function(self, genotype: Genotype, chromosome: str, idx: int):
        self.decoder.adapter.set_function(
            genotype, chromosome, idx, self.random_function(chromosome)
        )

    def mutate_connections(
        self,
        genotype: Genotype,
        chromosome: str,
        idx: int,
        only_one: int = None,
    ):
        new_edges = self.random_edges(idx, chromosome)
        if only_one is not None:
            new_value = new_edges[only_one]
            new_edges = self.decoder.adapter.get_edges(
                genotype, chromosome, idx
            )
            new_edges[only_one] = new_value
        self.decoder.adapter.set_edges(genotype, chromosome, idx, new_edges)

    def mutate_parameters(
        self,
        genotype: Genotype,
        chromosome: str,
        idx: int,
        only_one: int = None,
    ):
        new_random_parameters = self.random_parameters(chromosome)
        old_parameters = self.decoder.adapter.get_parameters(
            genotype, chromosome, idx
        )
        new_parameters = self.effect.call(
            old_parameters, new_random_parameters
        )
        if only_one is not None:

            old_parameters[only_one] = new_parameters[only_one]
            new_parameters = old_parameters.copy()
        self.decoder.adapter.set_parameters(
            genotype, chromosome, idx, new_parameters
        )

    def mutate_output(self, genotype: Genotype, idx: int):
        self.decoder.adapter.set_output(genotype, idx, self.random_output)

    @abstractmethod
    def mutate(self, genotype: Genotype):
        pass
