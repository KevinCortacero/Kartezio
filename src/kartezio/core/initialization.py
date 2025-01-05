from typing import Dict

from kartezio.core.components import Genotype, Initialization, register
from kartezio.mutation.base import PointMutation


@register(Initialization)
class CopyGenotype(Initialization):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "CopyGenotype":
        pass

    def __init__(self, genotype, shape, n_functions):
        super().__init__(shape, n_functions)
        self.genotype = genotype

    def mutate(self, genotype):
        return self.genotype.clone()


@register(Initialization)
class RandomInit(Initialization):
    """
    Can be used to initialize genotype randomly
    """

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "RandomInit":
        pass

    def __init__(self, mutation):
        super().__init__()
        self.mutation = mutation

    def mutate(self, genotype: Genotype):
        # mutate genes
        for chromosome in self.mutation.adapter.chromosomes_infos.keys():
            for node in range(self.mutation.adapter.n_nodes):
                self.mutation.mutate_function(genotype, chromosome, node)
                self.mutation.mutate_edges(genotype, chromosome, node)
                self.mutation.mutate_parameters(genotype, chromosome, node)
        # mutate outputs
        for output in range(self.mutation.adapter.n_outputs):
            self.mutation.mutate_output(genotype, output)
        return genotype

    def random(self):
        genotype = self.mutation.adapter.new_genotype()
        return self.mutate(genotype)
