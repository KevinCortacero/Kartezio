from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from kartezio.components.components import Component, register
from kartezio.components.genotype import Genotype
from kartezio.evolution.decoder import Adapter
from kartezio.mutation.edges import MutationEdgesNormal, MutationEdgesUniform
from kartezio.mutation.effect import MutationUniform


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
            for node, mutation_parameter_index in np.transpose(
                sampling_indices
            ):
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
