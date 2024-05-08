from kartezio.core.strategy import Strategy
from kartezio.population import PopulationWithElite


class OnePlusLambda(Strategy):
    def __init__(self, init, mutation_system):
        self.n_parents = 1
        self.n_children = 4
        self.mutation_system = mutation_system
        self.fn_init = init

    def create_population(self, n_children):
        self.n_children = n_children
        population = PopulationWithElite(self.n_children)
        for i in range(population.size):
            individual = self.fn_init.random()
            population.individuals[i] = individual
        return population

    def selection(self, population: PopulationWithElite):
        return population.promote_new_parent()

    def reproduction(self, population: PopulationWithElite):
        elite = population.get_elite()
        for i in range(self.n_parents, population.size):
            population.individuals[i] = self.mutation_system.mutate(
                elite.clone()
            )
