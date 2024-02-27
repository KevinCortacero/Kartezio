from pprint import pprint
from typing import Dict

import numpy as np

from kartezio.core.components.base import register
from kartezio.core.population import Population


class IndividualHistory:
    def __init__(self):
        self.fitness = 0.0
        self.time = 0.0
        self.genotype = None


class PopulationHistory:
    def __init__(self, n_individuals):
        self.individuals = {}
        for i in range(n_individuals):
            self.individuals[i] = IndividualHistory()

    def get_best_fitness(self):
        return (
            self.individuals[0],
            self.individuals[0].fitness,
            self.individuals[0].time,
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
        return self.individuals[0]

    def promote_new_parent(self):
        changed = False
        fitness = self.get_fitness()
        times = self.get_time()
        # best_fitness_idx = np.argsort(self.get_score())[0]
        best_fitness_idx = np.argsort(self.get_score())[0]
        if best_fitness_idx != 0:
            changed = True
            self.individuals[0] = self.individuals[best_fitness_idx].clone()
            self._fitness["fitness"][0] = fitness[best_fitness_idx]
            self._fitness["time"][0] = times[best_fitness_idx]


        state = PopulationHistory(self.size)

        for i in range(len(self.individuals)):
            state.individuals[i].genotype = self.individuals[i].clone()
            state.individuals[i].fitness = fitness[i]
            state.individuals[i].time = times[i]
        return changed, state

    """
        def get_state(self):
        state = PopulationHistory(self.size)
        fitness = self.get_fitness()
        times = self.get_time()
        for i in range(len(self.individuals)):
            state.individuals[i].chromosome = self.individuals[i].chromosome
            state.individuals[i].outputs = self.individuals[i].outputs
            state.individuals[i].fitness = fitness[i]
            state.individuals[i].time = times[i]
        return state
    """

