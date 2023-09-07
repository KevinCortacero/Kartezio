from kartezio.model.evolution import KartezioES
from kartezio.population import PopulationWithElite


class OnePlusLambda(KartezioES):
    def __init__(self, _lambda, factory, init_method, mutation_method, fitness):
        self._mu = 1
        self._lambda = _lambda
        self.factory = factory
        self.init_method = init_method
        self.mutation_method = mutation_method
        self.fitness = fitness
        self.population = PopulationWithElite(_lambda)

    @property
    def elite(self):
        return self.population.get_elite()

    def initialization(self):
        for i in range(self.population.size):
            individual = self.init_method.mutate(self.factory.create())
            self.population[i] = individual

    def selection(self):
        new_elite, fitness = self.population.get_best_individual()
        self.population.set_elite(new_elite)

    def reproduction(self):
        elite = self.population.get_elite()
        for i in range(self._mu, self.population.size):
            self.population[i] = elite.clone()

    def mutation(self):
        for i in range(self._mu, self.population.size):
            self.population[i] = self.mutation_method.mutate(self.population[i])

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.call(y_true, y_pred)
        self.population.set_fitness(fitness)
