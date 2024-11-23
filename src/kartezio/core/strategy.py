from abc import ABC, abstractmethod
import numpy as np
from kartezio.core.population import Population, PopulationWithElite


class Strategy(ABC):
    @abstractmethod
    def selection(self, population: Population):
        pass

    @abstractmethod
    def reproduction(self, population: Population):
        pass


class OnePlusLambda(Strategy):
    def __init__(self, init, mutation_system, gamma=None, required_fps=None):
        self.n_parents = 1
        self.n_children = 4
        self.mutation_system = mutation_system
        self.fn_init = init
        self.gamma = None
        if required_fps is not None:
            self.required_fps = 1.0 / required_fps
        else:
            self.required_fps = None

    def create_population(self, n_children):
        self.n_children = n_children
        population = PopulationWithElite(self.n_children)
        for i in range(population.size):
            individual = self.fn_init.random()
            population.individuals[i] = individual
        return population

    def selection(self, population: PopulationWithElite):
        # apply random noise on raw fitness
        fitness = population.get_raw()
        if self.gamma is not None:
            noise_shape = fitness[0].shape
            noise = np.random.normal(1.0, self.gamma, noise_shape)
            # apply noise to fitness
            fitness = fitness * noise
        population.score.fitness = np.mean(fitness, axis=1)
        if self.required_fps:
            for i in range(population.size):
                if population.score.time[i] > self.required_fps:
                    population.score.fitness[i] = np.inf
        return population.promote_new_parent()

    def reproduction(self, population: PopulationWithElite):
        elite = population.get_elite()
        for i in range(self.n_parents, population.size):
            population.individuals[i] = self.mutation_system.mutate(
                elite.clone()
            )