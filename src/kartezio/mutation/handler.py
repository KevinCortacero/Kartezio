from kartezio.components.genotype import Genotype
from kartezio.core.decoder import Adapter
from kartezio.mutation.base import MutationRandom
from kartezio.mutation.behavioral import MutationBehavior
from kartezio.mutation.decay import MutationDecay


class MutationHandler:
    def __init__(self, adapter: Adapter):
        self.mutation = MutationRandom(adapter)
        self.behavior = None
        self.decay = None
        self.node_rate = None
        self.out_rate = None

    def set_behavior(self, behavior: MutationBehavior):
        self.behavior = behavior

    def set_decay(self, decay: MutationDecay):
        self.decay = decay

    def set_effect(self, effect):
        self.mutation.effect = effect

    def set_mutation_rates(self, node_rate, out_rate):
        self.node_rate = node_rate
        self.out_rate = out_rate

    def compile(self, n_iterations: int):
        self.mutation.node_rate = self.node_rate
        self.mutation.out_rate = self.out_rate
        self.mutation.effect.compile(n_iterations)
        if self.behavior:
            self.behavior.set_mutation(self.mutation)
        if self.decay:
            self.decay.set_mutation(self.mutation)
            self.decay.compile(n_iterations)

    def mutate(self, genotype: Genotype):
        if self.behavior:
            return self.behavior.mutate(genotype)
        else:
            return self.mutation.mutate(genotype)
