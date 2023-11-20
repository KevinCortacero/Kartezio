from kartezio.model.components import Decoder
from kartezio.model.evolution import KStrategy
from kartezio.mutation import MutationAllRandom, MutationClassic
from kartezio.population import PopulationWithElite


class OnePlusLambda(KStrategy):
    def __init__(self, decoder: Decoder):
        self.n_parents = 1
        self.n_children = 4
        self.fn_init = MutationAllRandom(decoder.infos, decoder.library.size)
        self.fn_mutation = MutationClassic(decoder.infos, decoder.library.size, 0.1, 0.1)

    @property
    def elite(self):
        return self.population.get_elite()

    def create_population(self, n_children):
        self.n_children = n_children
        population = PopulationWithElite(self.n_children)
        for i in range(population.size):
            individual = self.fn_init.random()
            population[i] = individual
        return population

    def selection(self):
        new_elite, fitness = self.population.get_best_individual()
        self.population.set_elite(new_elite)

    def reproduction(self):
        elite = self.population.get_elite()
        for i in range(self.n_parents, self.population.size):
            self.population[i] = elite.clone()

    def mutation(self):
        for i in range(self.n_parents, self.population.size):
            self.population[i] = self.mutation_method.mutate(self.population[i])

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.batch(y_true, y_pred)
        self.population.set_fitness(fitness)
