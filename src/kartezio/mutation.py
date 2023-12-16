import random
from typing import Dict

import numpy as np

from kartezio.core.components.base import register
from kartezio.core.components.genotype import Genotype
from kartezio.core.mutation.base import Mutation


@register(Mutation, "random")
class MutationRandom(Mutation):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Mutation":
        pass

    def __init__(self, template, n_functions: int, node_rate: float, out_rate: float):
        super().__init__(template, n_functions)
        self.node_rate = node_rate
        self.out_rate = out_rate
        self.n_mutations = int(
            np.floor(self.infos.n_nodes * self.infos.w * self.node_rate)
        )
        self.all_indices = np.indices((self.infos.n_nodes, self.infos.w))
        self.all_indices = np.vstack(
            (self.all_indices[0].ravel(), self.all_indices[1].ravel())
        ).T
        self.sampling_range = range(len(self.all_indices))

    def mutate(self, genotype: Genotype) -> Genotype:
        sampling_indices = np.random.choice(
            self.sampling_range, self.n_mutations, replace=False
        )
        sampling_indices = self.all_indices[sampling_indices]

        for idx, mutation_parameter_index in sampling_indices:
            if mutation_parameter_index == 0:
                self.mutate_function(genotype, idx)
            elif mutation_parameter_index <= self.infos.n_connections:
                connection_idx = mutation_parameter_index - 1
                self.mutate_connections(genotype, idx, only_one=connection_idx)
            else:
                parameter_idx = mutation_parameter_index - self.infos.n_connections - 1
                self.mutate_parameters(genotype, idx, only_one=parameter_idx)
        for output in range(self.infos.n_outputs):
            if random.random() < self.out_rate:
                self.mutate_output(genotype, output)
        return genotype
