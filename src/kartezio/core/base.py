from abc import ABC, abstractmethod
from typing import List

from kartezio.callback import Callback, Event
from kartezio.core.components.decoder import Decoder
from kartezio.core.components.endpoint import Endpoint
from kartezio.core.components.initialization import MutationAllRandom
from kartezio.core.components.library import Library
from kartezio.core.evolution import Fitness
from kartezio.core.helpers import Observable
from kartezio.core.mutation.base import Mutation
from kartezio.core.mutation.behavior import AccumulateBehavior, MutationBehavior
from kartezio.core.mutation.decay import (
    FactorDecay,
    LinearDecay,
    MutationDecay,
)
from kartezio.export import GenomeToPython
from kartezio.mutation import MutationRandom
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

    def set_mutation_rate(self, rate: float):
        self.strategy.fn_mutation.node_rate = rate

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
        print(fitness)
        self.population.set_fitness(fitness)

    def next(self):
        self.current_generation += 1


class ValidModel(ModelML, Observable):
    def __init__(self, decoder: Decoder, ga: GeneticAlgorithm):
        super().__init__()
        self.decoder = decoder
        self.ga = ga

    def evaluation(self, x, y):
        y_pred = self.decoder.decode_population(self.ga.population, x)
        self.ga.evaluation(y, y_pred)

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
        python_writer = GenomeToPython(self.decoder)
        python_writer.to_python_class(class_name, self.ga.population.get_elite())


class ModelDraft:
    class MutationSystem:
        def __init__(self, mutation: Mutation):
            self.mutation = mutation
            self.behavior = None
            self.decay = None

        def set_behavior(self, behavior: MutationBehavior):
            self.behavior = behavior

        def set_decay(self, decay: MutationDecay):
            self.decay = decay

        def compile(self):
            self.behavior.set_mutation(self.mutation)
            self.decay.set_mutation(self.mutation)

    def __init__(self, decoder: Decoder, fitness: Fitness):
        super().__init__()
        self.decoder = decoder
        self.init = MutationAllRandom(decoder)
        self.mutation = self.MutationSystem(MutationRandom(decoder, 0.15, 0.2))
        self.mutation.set_behavior(AccumulateBehavior(decoder))
        # self.mutation.set_decay(FactorDecay(0.9999))
        # self.mutation.set_decay(LinearDecay((0.15 - 0.05) / 200.0))
        # self.decay = LinearDecay(self.mutation, 0.2, 0.05, 200)
        self.fitness = fitness
        self.updatable = []

    def set_endpoint(self, endpoint: Endpoint):
        self.decoder.endpoint = endpoint

    def set_library(self, library: Library):
        self.decoder.library = library

    def set_decay(self, decay: MutationDecay):
        self.mutation.set_decay(decay)

    def compile(self, n_generations: int, n_children: int, callbacks: List[Callback]):
        self.mutation.compile()
        self.updatable.append(self.mutation.decay)
        ga = GeneticAlgorithm(self.decoder, self.fitness)
        ga.init(n_generations, n_children)
        model = ValidModel(self.decoder, ga)
        for updatable in self.updatable:
            model.attach(updatable)
        for callback in callbacks:
            callback.set_decoder(model.decoder)
            model.attach(callback)
        return model
