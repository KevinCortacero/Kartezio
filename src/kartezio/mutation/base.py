from typing import Dict

import numpy as np

from kartezio.core.components import Genotype, Mutation, register
from kartezio.evolution.decoder import Adapter


@register(Mutation)
class PointMutation(Mutation):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "PointMutation":
        pass

    def __init__(self, adapter: Adapter):
        super().__init__(adapter)
        self.node_rate = None
        self.out_rate = None

    def mutate(self, genotype: Genotype) -> Genotype:
        for c_nb in range(self.adapter.nb_chromosomes):
            chromosome = f"chromosome_{c_nb}"
            for (
                sequence,
                sequence_infos,
            ) in self.adapter.chromosomes_infos.items():
                random_matrix = np.random.random(size=sequence_infos.shape)
                sampling_indices = np.nonzero(random_matrix < self.node_rate)
                for node, mutation_parameter_index in np.transpose(sampling_indices):
                    if mutation_parameter_index == 0:
                        self.mutate_function(genotype, chromosome, sequence, node)
                    elif mutation_parameter_index <= sequence_infos.n_edges:
                        connection_idx = mutation_parameter_index - 1
                        self.mutate_edges(
                            genotype,
                            chromosome,
                            sequence,
                            node,
                            only_one=connection_idx,
                        )
                    else:
                        parameter_idx = (
                            mutation_parameter_index - sequence_infos.n_edges - 1
                        )
                        self.mutate_parameters(
                            genotype,
                            chromosome,
                            sequence,
                            node,
                            only_one=parameter_idx,
                        )
            random_matrix = np.random.random(size=self.adapter.n_outputs)
            sampling_indices = np.nonzero(random_matrix < self.out_rate)
            for output in sampling_indices:
                self.mutate_output(genotype, chromosome, output)
        return genotype
