from abc import ABC
from typing import Dict

from kartezio.core.components.base import register
from kartezio.core.components.decoder import DecoderPoly
from kartezio.core.components.genotype import MultiChromosomes
from kartezio.core.mutation.base import Mutation, MutationPoly


@register(Mutation, "all_random")
class MutationAllRandom(Mutation):
    """
    Can be used to initialize genome (genome) randomly
    """

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Mutation":
        pass

    def __init__(self, decoder):
        super().__init__(decoder)

    def mutate(self, genotype):
        # mutate genes
        for i in range(self.decoder.adapter.n_nodes):
            self.mutate_function(genotype, i)
            self.mutate_connections(genotype, i)
            self.mutate_parameters(genotype, i)
        # mutate outputs
        for i in range(self.decoder.adapter.n_outputs):
            self.mutate_output(genotype, i)
        return genotype

    def random(self):
        genotype = self.decoder.adapter.new()
        return self.mutate(genotype)


@register(Mutation, "copy")
class CopyGenotype(Mutation):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Component":
        pass

    def __init__(self, genotype, shape, n_functions):
        super().__init__(shape, n_functions)
        self.genotype = genotype

    def mutate(self, genotype):
        return self.genotype.clone()


@register(Mutation, "all_random_poly")
class MutationAllRandomPoly(MutationPoly):
    """
    Can be used to initialize genome (genome) randomly
    """

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "MutationAllRandomPoly":
        pass

    def __init__(self, decoder: DecoderPoly):
        super().__init__(decoder)

    def mutate(self, genotype: MultiChromosomes):
        # mutate genes
        for t in range(len(self.decoder.adapter.types_map)):
            for n in range(self.decoder.adapter.n_nodes):
                self.mutate_function(genotype, t, n)
                self.mutate_connections(genotype, t, n)
                self.mutate_parameters(genotype, t, n)
        # mutate outputs
        for o in range(self.decoder.adapter.n_outputs):
            self.mutate_output(genotype, o)
        return genotype

    def random(self):
        genotype = self.decoder.adapter.new()
        return self.mutate(genotype)
