from kartezio.core.decoder import Decoder
from kartezio.components.genotype import Genotype
from kartezio.mutation.base import Mutation


class MutationBehavior:
    def __init__(self):
        self._mutation = None

    def mutate(self, genotype: Genotype):
        return self._mutation.mutate(genotype)

    def set_mutation(self, mutation: Mutation):
        self._mutation = mutation


class AccumulateBehavior(MutationBehavior):
    def __init__(self, decoder: Decoder):
        super().__init__()
        self.decoder = decoder

    def mutate(self, genotype: Genotype):
        changed = False
        active_nodes = self.decoder.parse_to_graphs(genotype)
        while not changed:
            genotype = self._mutation.mutate(genotype)
            new_active_nodes = self.decoder.parse_to_graphs(genotype)
            changed = active_nodes != new_active_nodes
        return genotype
