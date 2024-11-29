from abc import ABC
from typing import Dict

from kartezio.components.base import register
from kartezio.components.genotype import Genotype
from kartezio.core.decoder import Adapter
from kartezio.mutation.base import Mutation

"""
@register(Mutation, "all_random")
class RandomInit(Mutation):
    '''
    Can be used to initialize genome (genome) randomly
    '''

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Mutation":
        pass

    def __init__(self, adapter: Adapter):
        super().__init__(adapter)

    def mutate(self, genotype):
        # mutate genes
        for i in range(self.adapter.n_nodes):
            self.mutate_function(genotype, i)
            self.mutate_connections(genotype, i)
            self.mutate_parameters(genotype, i)
        # mutate outputs
        for i in range(self.adapter.n_outputs):
            self.mutate_output(genotype, i)
        return genotype

    def random(self):
        genotype = self.adapter.new_genotype()
        return self.mutate(genotype)

"""


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
class RandomInit(Mutation):
    """
    Can be used to initialize genome (genome) randomly
    """

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "MutationAllRandomPoly":
        pass

    def __init__(self, adapter: Adapter):
        super().__init__(adapter)

    def mutate(self, genotype: Genotype):
        # mutate genes
        for chromosome in self.adapter.chromosomes_infos.keys():
            for node in range(self.adapter.n_nodes):
                self.mutate_function(genotype, chromosome, node)
                self.mutate_connections(genotype, chromosome, node)
                self.mutate_parameters(genotype, chromosome, node)
        # mutate outputs
        for output in range(self.adapter.n_outputs):
            self.mutate_output(genotype, output)
        return genotype

    def random(self):
        genotype = self.adapter.new_genotype()
        return self.mutate(genotype)
