import random

import numpy as np

from kartezio.model.components import GenomeShape, KartezioGenome
from kartezio.model.evolution import KartezioMutation
from kartezio.model.registry import registry


class GoldmanWrapper(KartezioMutation):
    def __init__(self, mutation, decoder):
        super().__init__(None, None)
        self.mutation = mutation
        self.parser = decoder

    def mutate(self, genome):
        changed = False
        active_nodes = self.parser.parse_to_graphs(genome)
        while not changed:
            genome = self.mutation.mutate(genome)
            new_active_nodes = self.parser.parse_to_graphs(genome)
            changed = active_nodes != new_active_nodes
        return genome


@registry.mutations.add("classic")
class MutationClassic(KartezioMutation):
    def __init__(self, shape, n_functions, mutation_rate, output_mutation_rate):
        super().__init__(shape, n_functions)
        self.mutation_rate = mutation_rate
        self.output_mutation_rate = output_mutation_rate
        self.n_mutations = int(
            np.floor(self.shape.primitives * self.shape.w * self.mutation_rate)
        )
        self.all_indices = np.indices((self.shape.primitives, self.shape.w))
        self.all_indices = np.vstack(
            (self.all_indices[0].ravel(), self.all_indices[1].ravel())
        ).T
        self.sampling_range = range(len(self.all_indices))

    def mutate(self, genome):
        sampling_indices = np.random.choice(
            self.sampling_range, self.n_mutations, replace=False
        )
        sampling_indices = self.all_indices[sampling_indices]

        for idx, mutation_parameter_index in sampling_indices:
            if mutation_parameter_index == 0:
                self.mutate_function(genome, idx)
            elif mutation_parameter_index <= self.shape.connections:
                connection_idx = mutation_parameter_index - 1
                self.mutate_connections(genome, idx, only_one=connection_idx)
            else:
                parameter_idx = mutation_parameter_index - self.shape.connections - 1
                self.mutate_parameters(genome, idx, only_one=parameter_idx)
        for output in range(self.shape.outputs):
            if random.random() < self.output_mutation_rate:
                self.mutate_output(genome, output)
        return genome


@registry.mutations.add("all_random")
class MutationAllRandom(KartezioMutation):
    """
    Can be used to initialize genome (genome) randomly
    """

    def __init__(self, metadata: GenomeShape, n_functions: int):
        super().__init__(metadata, n_functions)

    def mutate(self, genome: KartezioGenome):
        # mutate genes
        for i in range(self.shape.primitives):
            self.mutate_function(genome, i)
            self.mutate_connections(genome, i)
            self.mutate_parameters(genome, i)
        # mutate outputs
        for i in range(self.shape.outputs):
            self.mutate_output(genome, i)
        return genome


@registry.mutations.add("copy")
class CopyGenome:
    def __init__(self, genome: KartezioGenome):
        self.genome = genome

    def mutate(self, _genome: KartezioGenome):
        return self.genome.clone()
