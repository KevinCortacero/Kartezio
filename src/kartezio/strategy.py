from kartezio.core.components.decoder import Decoder
from kartezio.core.components.initialization import MutationAllRandom
from kartezio.core.strategy import Strategy
from kartezio.mutation import MutationRandom
from kartezio.population import PopulationWithElite


class OnePlusLambda(Strategy):
    def __init__(self, decoder: Decoder):
        self.n_parents = 1
        self.n_children = 4
        self.fn_init = MutationAllRandom(decoder.infos, decoder.library.size)
        self.fn_mutation = MutationRandom(decoder.infos, decoder.library.size, 0.1, 0.1)

    @property
    def get_elite(self, population: PopulationWithElite):
        return population.get_elite()

    def create_population(self, n_children):
        self.n_children = n_children
        population = PopulationWithElite(self.n_children)
        for i in range(population.size):
            individual = self.fn_init.random()
            population[i] = individual
        return population

    def selection(self, population: PopulationWithElite):
        population.promote_new_parent()

    def reproduction(self, population: PopulationWithElite):
        elite = population.get_elite()
        for i in range(self.n_parents, population.size):
            population[i] = self.fn_mutation.mutate(elite.clone())

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.batch(y_true, y_pred)
        self.population.set_fitness(fitness)
