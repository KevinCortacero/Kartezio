from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from kartezio.components.core import Component, register
from kartezio.components.genotype import Genotype
from kartezio.evolution.decoder import Adapter
from kartezio.mutation.edges import MutationEdgesNormal, MutationEdgesUniform
from kartezio.mutation.effect import MutationUniform


class Mutation(Component, ABC):
    def __init__(self, adapter: Adapter):
        super().__init__()
        self.adapter = adapter
        self.parameter_max_value = 256
        self.effect = MutationUniform()
        self.edges_weights = MutationEdgesUniform()

    def random_parameters(self, chromosome: int):
        return np.random.randint(
            self.parameter_max_value,
            size=self.adapter.chromosomes_infos[chromosome].n_parameters,
        )

    def random_function(self, chromosome: str):
        return np.random.randint(self.adapter.chromosomes_infos[chromosome].n_functions)

    def mutate_function(self, genotype: Genotype, chromosome: str, idx: int):
        self.adapter.set_function(
            genotype, chromosome, idx, self.random_function(chromosome)
        )

    def mutate_edges(
        self,
        genotype: Genotype,
        chromosome: str,
        idx: int,
        only_one: int = None,
    ):
        p = self.edges_weights.weights_edges(self.adapter.n_inputs + idx)
        print(idx, p)
        new_edges = np.random.choice(
            range(self.adapter.n_inputs + idx),
            size=self.adapter.chromosomes_infos[chromosome].n_edges,
            p=p,
        )
        if only_one is not None:
            new_value = new_edges[only_one]
            new_edges = self.adapter.get_edges(genotype, chromosome, idx)
            new_edges[only_one] = new_value
        self.adapter.set_edges(genotype, chromosome, idx, new_edges)

    def mutate_parameters(
        self,
        genotype: Genotype,
        chromosome: str,
        idx: int,
        only_one: int = None,
    ):
        new_random_parameters = self.random_parameters(chromosome)
        old_parameters = self.adapter.get_parameters(genotype, chromosome, idx)
        new_parameters = self.effect.call(old_parameters, new_random_parameters)
        if only_one is not None:
            old_parameters[only_one] = new_parameters[only_one]
            new_parameters = old_parameters.copy()
        self.adapter.set_parameters(genotype, chromosome, idx, new_parameters)

    def mutate_output(self, genotype: Genotype, idx: int):
        p = self.edges_weights.weights_edges(self.adapter.out_idx)
        new_edges = np.random.choice(range(self.adapter.out_idx), size=1, p=p)
        self.adapter.set_output(genotype, idx, new_edges)

    @abstractmethod
    def mutate(self, genotype: Genotype):
        pass


@register(Mutation, "random")
class MutationRandom(Mutation):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "MutationRandom":
        pass

    def __init__(self, adapter: Adapter):
        super().__init__(adapter)
        self.node_rate = None
        self.out_rate = None

    def mutate(self, genotype: Genotype) -> Genotype:
        for (
            chromosome,
            chromosome_infos,
        ) in self.adapter.chromosomes_infos.items():
            random_matrix = np.random.random(size=chromosome_infos.shape)
            sampling_indices = np.nonzero(random_matrix < self.node_rate)
            for node, mutation_parameter_index in np.transpose(sampling_indices):
                if mutation_parameter_index == 0:
                    self.mutate_function(genotype, chromosome, node)
                elif mutation_parameter_index <= chromosome_infos.n_edges:
                    connection_idx = mutation_parameter_index - 1
                    self.mutate_edges(
                        genotype, chromosome, node, only_one=connection_idx
                    )
                else:
                    parameter_idx = (
                        mutation_parameter_index - chromosome_infos.n_edges - 1
                    )
                    self.mutate_parameters(
                        genotype, chromosome, node, only_one=parameter_idx
                    )
        random_matrix = np.random.random(size=self.adapter.n_outputs)
        sampling_indices = np.nonzero(random_matrix < self.out_rate)
        for output in sampling_indices:
            self.mutate_output(genotype, output)
        return genotype
