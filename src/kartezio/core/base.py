from abc import ABC, abstractmethod
from typing import List

from kartezio.callback import Callback, Event
from kartezio.components.endpoint import Endpoint
from kartezio.components.genotype import Genotype
from kartezio.components.initializer import RandomInit
from kartezio.components.library import Library
from kartezio.core.decoder import DecoderPoly
from kartezio.core.evolution import Fitness
from kartezio.core.helpers import Observable
from kartezio.core.strategy import OnePlusLambda, Strategy
from kartezio.export import PythonClassWriter
from kartezio.mutation.behavioral import MutationBehavior
from kartezio.mutation.decay import MutationDecay
from kartezio.mutation.handler import MutationHandler


class GenericModel(ABC):
    @abstractmethod
    def predict(self, x: List):
        pass


class ModelTrainer(GenericModel):
    @abstractmethod
    def fit(self, x: List, y: List):
        pass


class GeneticAlgorithm:
    def __init__(self, adapter, fitness: Fitness):
        self.strategy = OnePlusLambda(adapter)
        self.population = None
        self.fitness = fitness
        self.current_iteration = 0
        self.n_iterations = 0

    def compile(self, n_iterations: int, n_children: int):
        self.strategy.compile(n_iterations)
        self.current_generation = 0
        self.n_iterations = n_iterations
        self.population = self.strategy.create_population(n_children)

    def is_satisfying(self):
        return (
            self.current_generation >= self.n_iterations
            or self.population.get_fitness()[0] == 0.0
        )

    def selection(self):
        return self.strategy.selection(self.population)

    def reproduction(self):
        self.strategy.reproduction(self.population)

    def evaluation(self, y_true, y_pred):
        fitness = self.fitness.batch(y_true, y_pred, reduction="raw")
        self.population.set_raw_fitness(fitness)

    def next(self):
        self.current_generation += 1


class CartesiaGeneticProgramming(ModelTrainer, Observable):
    def __init__(self, n_inputs, n_nodes, libraries, endpoint, fitness):
        super().__init__()
        self.decoder = DecoderPoly(n_inputs, n_nodes, libraries, endpoint)
        self.genetic_algorithm = GeneticAlgorithm(
            self.decoder.adapter, fitness
        )

    @property
    def population(self):
        return self.genetic_algorithm.population

    def evaluation(self, x, y):
        y_pred = self.decoder.decode_population(self.population, x)
        self.genetic_algorithm.evaluation(y, y_pred)

    def fit(
        self,
        x,
        y,
    ):
        self.evaluation(x, y)
        changed, state = self.genetic_algorithm.selection()
        if changed:
            self.send_event(Event.Events.NEW_PARENT, state, force=True)
        self.send_event(Event.Events.START_LOOP, state, force=True)
        while not self.genetic_algorithm.is_satisfying():
            self.send_event(Event.Events.START_STEP, state)
            self.genetic_algorithm.reproduction()
            self.evaluation(x, y)
            changed, state = self.genetic_algorithm.selection()
            if changed:
                self.send_event(Event.Events.NEW_PARENT, state, force=True)
            self.send_event(Event.Events.END_STEP, state)
            self.genetic_algorithm.next()
        self.send_event(Event.Events.END_LOOP, state, force=True)
        elite = self.population.get_elite()
        return elite, state

    def send_event(self, name, state, force=False):
        event = Event(
            self.genetic_algorithm.current_generation, name, state, force
        )
        self.notify(event)

    def evaluate(self, x, y):
        y_pred, t = self.predict(x)
        return self.genetic_algorithm.fitness.batch(y, [y_pred])

    def predict(self, x):
        return self.decoder.decode(self.population.get_elite(), x)

    def print_python_class(self, class_name):
        python_writer = PythonClassWriter(self.decoder)
        python_writer.to_python_class(class_name, self.population.get_elite())

    def display_elite(self):
        elite = self.population.get_elite()
        print(elite[0])
        print(elite.outputs)


class ModelBuilder:
    def __init__(
        self,
        n_inputs: int,
        n_nodes: int,
        libraries: List[Library],
        endpoint: Endpoint,
        fitness: Fitness,
    ):
        super().__init__()
        if not isinstance(libraries, list):
            libraries = [libraries]
        self.model_trainer = CartesiaGeneticProgramming(
            n_inputs, n_nodes, libraries, endpoint, fitness
        )
        self.updatable = []

    @property
    def decoder(self) -> DecoderPoly:
        return self.model_trainer.decoder

    def set_endpoint(self, endpoint: Endpoint):
        self.decoder.endpoint = endpoint

    def set_library(self, library: Library):
        self.decoder.library = library

    @property
    def strategy(self) -> OnePlusLambda:
        return self.model_trainer.genetic_algorithm.strategy

    def set_required_fps(self, required_fps):
        self.strategy.set_required_fps(required_fps)

    def set_gamma(self, gamma):
        self.strategy.set_gamma(gamma)

    @property
    def mutation(self) -> MutationHandler:
        return self.strategy.mutation_handler

    def set_decay(self, decay: MutationDecay):
        self.mutation.set_decay(decay)

    def set_behavior(self, behavior: MutationBehavior):
        self.mutation.set_behavior(behavior)
        self.mutation.behavior.set_decoder(self.decoder)

    def set_mutation_rates(self, node_rate, out_rate):
        self.mutation.set_mutation_rates(node_rate, out_rate)

    def set_mutation_effect(self, effect):
        self.mutation.set_effect(effect)

    def compile(
        self, n_iterations: int, n_children: int, callbacks: List[Callback]
    ):
        # compile genetic algorithm
        self.model_trainer.genetic_algorithm.compile(n_iterations, n_children)
        self.updatable.append(self.mutation.mutation.effect)
        if self.mutation.decay:
            self.updatable.append(self.mutation.decay)
        for updatable in self.updatable:
            self.model_trainer.attach(updatable)
        for callback in callbacks:
            callback.set_decoder(self.model_trainer.decoder)
            self.model_trainer.attach(callback)

        return self.model_trainer
