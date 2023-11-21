from abc import ABC, abstractmethod
from typing import List

from kartezio.callback import Event
from kartezio.export import GenomeToPython
from kartezio.model.components import Decoder
from kartezio.model.evolution import Fitness
from kartezio.model.helpers import Observable
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
    def __init__(self, decoder: Decoder, fitness: Fitness):
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
        return (
            self.current_generation >= self.n_generations
            or self.population.fitness[0] == 0.0
        )

    def selection(self):
        self.strategy.selection(self.population)

    def reproduction(self):
        self.strategy.reproduction(self.population)

    def mutation(self):
        self.strategy.mutation()

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.batch(y_true, y_pred)
        self.population.set_fitness(fitness)

    def next(self):
        self.current_generation += 1


class ModelBase(ModelML, Observable):
    def __init__(self, decoder: Decoder, fitness: Fitness):
        super().__init__()
        self.decoder = decoder
        self.ga = GeneticAlgorithm(self.decoder, fitness)
        self.callbacks = []

    def evaluation(self, x, y):
        y_pred = self.decoder.decode_population(self.ga.population, x)
        self.ga.evaluation(y, y_pred)

    def compile(self, n_generations: int, n_children: int):
        self.ga.init(n_generations, n_children)

    def fit(
        self,
        x,
        y,
    ):
        self.evaluation(x, y)
        self.ga.selection()
        self.send_event(Event.START_LOOP, force=True)
        while not self.ga.is_satisfying():
            self.send_event(Event.START_STEP)
            self.ga.reproduction()
            self.evaluation(x, y)
            self.ga.selection()
            self.ga.next()
            self.send_event(Event.END_STEP)
        self.send_event(Event.END_LOOP, force=True)
        history = self.ga.population.history()
        elite = self.ga.population.get_elite()
        return elite, history

    def send_event(self, name, force=False):
        event = {
            "n": self.ga.current_generation,
            "name": name,
            "content": self.ga.population.history(),
            "force": force,
        }
        self.notify(event)

    def evaluate(self, x, y):
        y_pred, t = self.predict(x)
        return self.ga.fitness.batch(y, [y_pred])

    def predict(self, x):
        return self.decoder.decode(self.ga.population.get_elite(), x)

    def save_elite(self, filepath, dataset):
        JsonSaver(dataset, self.parser).save_individual(
            filepath, self.strategy.population.history().individuals[0]
        )

    def print_python_class(self, class_name):
        python_writer = GenomeToPython(self.parser)
        python_writer.to_python_class(class_name, self.strategy.elite)
