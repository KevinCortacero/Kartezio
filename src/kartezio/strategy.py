from kartezio.model.components import Decoder
from kartezio.model.strategy import Strategy
from kartezio.model.population import Population
from kartezio.mutation import MutationAllRandom, MutationClassic
from kartezio.population import PopulationWithElite


class OnePlusLambda(Strategy):
    def __init__(self, decoder: Decoder):
        self.n_parents = 1
        self.n_children = 4
        self.fn_init = MutationAllRandom(decoder.infos, decoder.library.size)
        self.fn_mutation = MutationClassic(
            decoder.infos, decoder.library.size, 0.1, 0.1
        )

    @property
    def get_elite(self, population: Population):
        return population.get_elite()

    def create_population(self, n_children):
        self.n_children = n_children
        population = PopulationWithElite(self.n_children)
        for i in range(population.size):
            individual = self.fn_init.random()
            population[i] = individual
        return population

    def selection(self, population: Population):
        new_elite, fitness = population.get_best_individual()
        population.set_elite(new_elite)

    def reproduction(self, population: Population):
        elite = population.get_elite()
        for i in range(self.n_parents, population.size):
            population[i] = self.fn_mutation.mutate(elite.clone())

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.batch(y_true, y_pred)
        self.population.set_fitness(fitness)
