from typing import Dict

import numpy as np

from kartezio.core.components.base import register
from kartezio.core.population import Population


class IndividualHistory:
    def __init__(self):
        self.fitness = {"fitness": 0.0, "time": 0.0}
        self.chromosome = None
        self.outputs = None

    def set_genes(self, chromosome, outputs):
        self.chromosome = chromosome
        self.outputs = outputs

    def set_values(self, chromosome, outputs, fitness, time):
        self.set_genes(chromosome, outputs)
        self.fitness["fitness"] = fitness
        self.fitness["time"] = time


class PopulationHistory:
    def __init__(self, n_individuals):
        self.individuals = {}
        for i in range(n_individuals):
            self.individuals[i] = IndividualHistory()

    def fill(self, individuals, fitness, times):
        for i in range(len(individuals)):
            self.individuals[i].set_values(
                individuals[i].chromosome,
                individuals[i].outputs,
                float(fitness[i]),
                float(times[i]),
            )

    def get_best_fitness(self):
        return (
            self.individuals[0].fitness["fitness"],
            self.individuals[0].fitness["time"],
        )

    def get_individuals(self):
        return self.individuals.items()


@register(Population, "one_elite")
class PopulationWithElite(Population):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Component":
        pass

    def __init__(self, n_children):
        super().__init__(1 + n_children)

    def get_elite(self):
        return self[0]

    def promote_new_parent(self):
        best_fitness_idx = np.argsort(self.score)[0]
        self[0] = self[best_fitness_idx]
        self._fitness["fitness"][0] = self.fitness[best_fitness_idx]
        self._fitness["time"][0] = self.time[best_fitness_idx]

    def history(self):
        population_history = PopulationHistory(self.size)
        population_history.fill(self.individuals, self.fitness, self.time)
        return population_history
