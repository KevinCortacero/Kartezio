from typing import Dict

import numpy as np

from kartezio.core.components.base import register
from kartezio.core.components.decoder import Decoder, DecoderPoly
from kartezio.core.components.genotype import Genotype
from kartezio.core.mutation.base import Mutation, MutationPoly


@register(Mutation, "random")
class MutationRandom(Mutation):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Mutation":
        pass

    def __init__(self, decoder: Decoder, node_rate: float, out_rate: float):
        super().__init__(decoder)
        self.node_rate = node_rate
        self.out_rate = out_rate
        self.nodes_shape = (decoder.adapter.n_nodes, decoder.adapter.w)
        self.outputs_shape = decoder.adapter.n_outputs

    def mutate(self, genotype: Genotype) -> Genotype:
        random_matrix = np.random.random(size=self.nodes_shape)
        sampling_indices = np.nonzero(random_matrix < self.node_rate)
        for idx, mutation_parameter_index in np.transpose(sampling_indices):
            if mutation_parameter_index == 0:
                self.mutate_function(genotype, idx)
            elif (
                mutation_parameter_index <= self.decoder.adapter.n_connections
            ):
                connection_idx = mutation_parameter_index - 1
                self.mutate_connections(genotype, idx, only_one=connection_idx)
            else:
                parameter_idx = (
                    mutation_parameter_index
                    - self.decoder.adapter.n_connections
                    - 1
                )
                self.mutate_parameters(genotype, idx, only_one=parameter_idx)

        random_matrix = np.random.random(size=self.outputs_shape)
        sampling_indices = np.nonzero(random_matrix < self.out_rate)
        for output in sampling_indices:
            self.mutate_output(genotype, output)
        return genotype


@register(Mutation, "random_poly")
class MutationRandomPoly(MutationPoly):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "MutationRandomPoly":
        pass

    def __init__(
        self, decoder: DecoderPoly, node_rate: float, out_rate: float
    ):
        super().__init__(decoder)
        self.node_rate = node_rate
        self.out_rate = out_rate
        self.outputs_shape = decoder.adapter.n_outputs

    def mutate(self, genotype: Genotype) -> Genotype:
        for (
            chromosome,
            chromosome_infos,
        ) in self.decoder.adapter.chromosomes_infos.items():
            random_matrix = np.random.random(size=chromosome_infos.shape)
            sampling_indices = np.nonzero(random_matrix < self.node_rate)
            for node, mutation_parameter_index in np.transpose(
                sampling_indices
            ):
                if mutation_parameter_index == 0:
                    self.mutate_function(genotype, chromosome, node)
                elif mutation_parameter_index <= chromosome_infos.n_edges:
                    connection_idx = mutation_parameter_index - 1
                    self.mutate_connections(
                        genotype, chromosome, node, only_one=connection_idx
                    )
                else:
                    parameter_idx = (
                        mutation_parameter_index - chromosome_infos.n_edges - 1
                    )
                    self.mutate_parameters(
                        genotype, chromosome, node, only_one=parameter_idx
                    )
        random_matrix = np.random.random(size=self.outputs_shape)
        sampling_indices = np.nonzero(random_matrix < self.out_rate)
        for output in sampling_indices:
            self.mutate_output(genotype, output)
        return genotype
