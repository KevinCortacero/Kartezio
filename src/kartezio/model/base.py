from abc import ABC, abstractmethod
from typing import List

from kartezio.callback import Event
from kartezio.export import GenomeToPython
from kartezio.model.components import Decoder, GenotypeInfos
from kartezio.model.evolution import KartezioFitness
from kartezio.model.helpers import Observable
from kartezio.population import PopulationWithElite
from kartezio.strategy import OnePlusLambda
from kartezio.utils.io import JsonSaver


class ModelML(ABC):
    @abstractmethod
    def fit(self, x: List, y: List):
        pass

    @abstractmethod
    def evaluate(self, x: List, y: List):
        pass

    @abstractmethod
    def predict(self, x: List):
        pass


class GeneticAlgorithm:
    def __init__(self, decoder: Decoder, fitness: KartezioFitness):
        self.strategy = OnePlusLambda(decoder)
        self.population = None
        self.fitness = fitness
        self.current_generation = 0
        self.n_generations = 0

    def init(self, n_generations: int, n_children: int):
        self.current_generation = 0
        self.n_generations = n_generations
        self.population = self.strategy.create_population(n_children)

    def update(self):
        pass


    def fit(self, x: List, y: List):
        pass

    def is_satisfying(self):
        end_of_generations = self.current_generation >= self.generations
        best_fitness_reached = self.strategy.population.fitness[0] == 0.0
        return end_of_generations or best_fitness_reached

    def selection(self):
        self.strategy.selection()

    def reproduction(self):
        self.strategy.reproduction()

    def mutation(self):
        self.strategy.mutation()

    def evaluation(self, y_true, y_pred):
        self.strategy.evaluation(y_true, y_pred)

    def next(self):
        self.current_generation += 1


class ModelBase(ModelML, Observable):
    def __init__(self, decoder: Decoder, fitness: KartezioFitness):
        super().__init__()
        self.decoder = decoder
        self.ga = GeneticAlgorithm(self.decoder, fitness)
        # self.strategy = OnePlusLambda()
        self.callbacks = []
        # self.parser = parser
        # self.generations = generations

    def compile(self, n_generations: int, n_children: int):
        self.ga.init(n_generations, n_children)

    def fit(
        self,
        x,
        y,
    ):
        genetic_algorithm = GeneticAlgorithm(self.strategy, self.generations)
        genetic_algorithm.initialization()
        y_pred = self.parser.parse_population(self.strategy.population, x)
        genetic_algorithm.evaluation(y, y_pred)
        self._notify(0, Event.START_LOOP, force=True)
        while not genetic_algorithm.is_satisfying():
            self._notify(genetic_algorithm.current_generation, Event.START_STEP)
            genetic_algorithm.selection()
            genetic_algorithm.reproduction()
            genetic_algorithm.mutation()
            y_pred = self.parser.parse_population(self.strategy.population, x)
            genetic_algorithm.evaluation(y, y_pred)
            genetic_algorithm.next()
            self._notify(genetic_algorithm.current_generation, Event.END_STEP)
        self._notify(genetic_algorithm.current_generation, Event.END_LOOP, force=True)
        history = self.strategy.population.history()
        elite = self.strategy.elite
        return elite, history

    def _notify(self, n, name, force=False):
        event = {
            "n": n,
            "name": name,
            "content": self.strategy.population.history(),
            "force": force,
        }
        self.notify(event)

    def evaluate(self, x, y):
        y_pred, t = self.predict(x)
        return self.strategy.fitness.compute(y, [y_pred])

    def predict(self, x):
        return self.parser.decode(self.strategy.elite, x)

    def save_elite(self, filepath, dataset):
        JsonSaver(dataset, self.parser).save_individual(
            filepath, self.strategy.population.history().individuals[0]
        )

    def print_python_class(self, class_name):
        python_writer = GenomeToPython(self.parser)
        python_writer.to_python_class(class_name, self.strategy.elite)
