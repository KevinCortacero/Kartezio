from kartezio.model.components import BaseGenotype, Decoder
from kartezio.model.mutation.base import Mutation


class MutationBehavior:
    def __init__(self, mutation: Mutation):
        self.__mutation = mutation

    def mutate(self, genotype: BaseGenotype):
        return self.__mutation.mutate(genotype)


class AccumulateBehavior(MutationBehavior):
    def __init__(self, mutation: Mutation, decoder: Decoder):
        super().__init__(mutation)
        self.decoder = decoder

    def mutate(self, genotype: BaseGenotype):
        changed = False
        active_nodes = self.decoder.parse_to_graphs(genotype)
        while not changed:
            genotype = self.__mutation.mutate(genotype)
            new_active_nodes = self.decoder.parse_to_graphs(genotype)
            changed = active_nodes != new_active_nodes
        return genotype
