from kartezio.core.components import Genotype, Initialization, register


@register(Initialization)
class CopyGenotype(Initialization):
    @classmethod
    def __from_dict__(cls, dict_infos: dict) -> "CopyGenotype":
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
    def __from_dict__(cls, dict_infos: dict) -> "RandomInit":
        pass

    def __init__(self, mutation):
        super().__init__()
        self.mutation = mutation

    def mutate(self, genotype: Genotype):
        # mutate genes
        for chromosome in genotype._chromosomes.keys():
            for sequence in self.mutation.adapter.chromosomes_infos.keys():
                for node in range(self.mutation.adapter.n_nodes):
                    self.mutation.mutate_function(
                        genotype, chromosome, sequence, node
                    )
                    self.mutation.mutate_edges(
                        genotype, chromosome, sequence, node
                    )
                    self.mutation.mutate_parameters(
                        genotype, chromosome, sequence, node
                    )
            # mutate outputs
            for output in range(self.mutation.adapter.n_outputs):
                self.mutation.mutate_output(genotype, chromosome, output)
        return genotype

    def random(self):
        genotype = self.mutation.adapter.new_genotype()
        return self.mutate(genotype)
