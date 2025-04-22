from kartezio.core.components import Genotype, UpdatableComponent
from kartezio.evolution.decoder import Adapter
from kartezio.mutation.base import PointMutation
from kartezio.mutation.behavioral import MutationBehavior
from kartezio.mutation.decay import MutationDecay
from kartezio.mutation.edges import MutationEdges, MutationEdgesUniform
from kartezio.mutation.effect import MutationEffect, MutationUniform


class MutationHandler:
    def __init__(self, adapter: Adapter):
        self.mutation = PointMutation(adapter)
        self.behavior = None
        self.decay = None
        self.effect = MutationUniform()
        self.edges_weights = MutationEdgesUniform()
        self.node_rate = None
        self.out_rate = None

    def set_behavior(self, behavior: MutationBehavior):
        self.behavior = behavior

    def set_decay(self, decay: MutationDecay):
        self.decay = decay

    def set_effect(self, effect: MutationEffect):
        self.effect = effect

    def set_edges(self, edges: MutationEdges):
        self.edges_weights = edges

    def set_mutation_rates(self, node_rate, out_rate):
        self.node_rate = node_rate
        self.out_rate = out_rate

    def compile(self, n_iterations: int):
        assert (
            self.node_rate is not None
        ), "Node rate must be set before compiling the mutation handler."
        assert (
            self.out_rate is not None
        ), "Output rate must be set before compiling the mutation handler."
        self.mutation.node_rate = self.node_rate
        self.mutation.out_rate = self.out_rate
        self.mutation.edges_weights = self.edges_weights
        self.mutation.parameters = self.effect
        self.mutation.parameters.compile(n_iterations)
        if self.behavior:
            self.behavior.set_mutation(self.mutation)
        if self.decay:
            self.decay.set_mutation(self.mutation)
            self.decay.compile(n_iterations)

    def collect_updatables(self):
        updatables = []
        if isinstance(self.mutation.parameters, UpdatableComponent):
            updatables.append(self.mutation.parameters)
        if self.behavior:
            if isinstance(self.behavior, UpdatableComponent):
                updatables.append(self.behavior)
        if self.decay:
            if isinstance(self.decay, UpdatableComponent):
                updatables.append(self.decay)
        return updatables

    def mutate(self, genotype: Genotype):
        if self.behavior:
            return self.behavior.mutate(genotype)
        else:
            return self.mutation.mutate(genotype)
