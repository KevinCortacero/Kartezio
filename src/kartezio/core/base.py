from abc import ABC, abstractmethod
from typing import List

from kartezio.callback import Callback, Event
from kartezio.core.decoder import Decoder
from kartezio.components.endpoint import Endpoint
from kartezio.components.genotype import Genotype
from kartezio.components.initializer import RandomInit
from kartezio.components.library import Library
from kartezio.core.evolution import Fitness
from kartezio.core.helpers import Observable
from kartezio.mutation.base import Mutation
from kartezio.mutation.behavioral import (
    AccumulateBehavior,
    MutationBehavior,
)
from kartezio.mutation.decay import MutationDecay
from kartezio.export import PythonClassWriter
from kartezio.core.strategy import OnePlusLambda


class GenericModel(ABC):
    @abstractmethod
    def predict(self, x: List):
        pass


class ModelTrainer(GenericModel):
    @abstractmethod
    def fit(self, x: List, y: List):
        pass


class GeneticAlgorithm:
    def __init__(self, init, mutation, fitness: Fitness, gamma=None, required_fps=None):
        self.strategy = OnePlusLambda(init, mutation, gamma, required_fps)
        self.population = None
        self.fitness = fitness
        self.current_iteration = 0
        self.n_iterations = 0

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
        changed, state = self.ga.selection()
        if changed:
            self.send_event(Event.Events.NEW_PARENT, state, force=True)
        self.send_event(Event.Events.START_LOOP, state, force=True)
        while not self.ga.is_satisfying():
            self.send_event(Event.Events.START_STEP, state)
            self.ga.reproduction()
            self.evaluation(x, y)
            changed, state = self.ga.selection()
            if changed:
                self.send_event(Event.Events.NEW_PARENT, state, force=True)
            self.send_event(Event.Events.END_STEP, state)
            self.ga.next()
        self.send_event(Event.Events.END_LOOP, state, force=True)
        elite = self.ga.population.get_elite()
        return elite, state

    def send_event(self, name, state, force=False):
        event = {
            "n": self.ga.current_generation,
            "name": name,
            "content": state,
            "force": force,
        }
        self.notify(event)

    def evaluate(self, x, y):
        y_pred, t = self.predict(x)
        return self.ga.fitness.batch(y, [y_pred])

    def predict(self, x):
        return self.decoder.decode(self.ga.population.get_elite(), x)

    def print_python_class(self, class_name):
        python_writer = PythonClassWriter(self.decoder)
        python_writer.to_python_class(
            class_name, self.ga.population.get_elite()
        )

    def display_elite(self):
        elite = self.ga.population.get_elite()
        print(elite[0])
        print(elite.outputs)


class ModelBuilder:
    class MutationSystem:
        def __init__(self, mutation: Mutation):
            self.mutation = mutation
            self.behavior = None
            self.decay = None
            self.node_rate = 0.05
            self.out_rate = 0.05

        def set_behavior(self, behavior: MutationBehavior):
            self.behavior = behavior

        def set_decay(self, decay: MutationDecay):
            self.decay = decay

        def set_effect(self, effect):
            self.mutation.effect = effect

        def set_mutation_rates(self, node_rate, out_rate):
            self.node_rate = node_rate
            self.out_rate = out_rate

        def compile(self, n_iterations: int):
            self.mutation.node_rate = self.node_rate
            self.mutation.out_rate = self.out_rate
            if self.behavior:
                self.behavior.set_mutation(self.mutation)
            if self.decay:
                self.decay.set_mutation(self.mutation)
                self.decay.compile(n_iterations)

        def mutate(self, genotype: Genotype):
            if self.behavior:
                return self.behavior.mutate(genotype)
            else:
                return self.mutation.mutate(genotype)

    def __init__(
        self,
        decoder: Decoder,
        fitness: Fitness,
        init=None,
        mutation=None,
        behavior=None,
    ):
        super().__init__()
        self.decoder = decoder
        if init:
            self.init = init
        else:
            self.init = MutationAllRandom(decoder)
        if mutation:
            self.mutation = self.MutationSystem(mutation)
        else:
            self.mutation = self.MutationSystem(
                MutationRandom(decoder, 0.05, 0.1)
            )
        if behavior == "accumulate":
            self.mutation.set_behavior(AccumulateBehavior(decoder))
        self.fitness = fitness
        self.updatable = []

    def set_endpoint(self, endpoint: Endpoint):
        self.decoder.endpoint = endpoint

    def set_library(self, library: Library):
        self.decoder.library = library

    def set_decay(self, decay: MutationDecay):
        self.mutation.set_decay(decay)

    def set_behavior(self, behavior: MutationBehavior):
        self.mutation.set_behavior(behavior)

    def set_mutation_rates(self, node_rate, out_rate):
        self.mutation.set_mutation_rates(node_rate, out_rate)

    def set_mutation_effect(self, effect):
        self.mutation.set_effect(effect)

    def compile(
        self, n_iterations: int, n_children: int, callbacks: List[Callback], gamma=None, required_fps=60
    ):
        self.mutation.compile(n_iterations)
        self.updatable.append(self.mutation.mutation.effect)
        if self.mutation.decay:
            self.updatable.append(self.mutation.decay)
        ga = GeneticAlgorithm(self.init, self.mutation, self.fitness, gamma, required_fps)
        ga.init(n_iterations, n_children)
        model_trainer = CartesiaGeneticProgramming(self.decoder, ga)
        for updatable in self.updatable:
            model_trainer.attach(updatable)
        for callback in callbacks:
            callback.set_decoder(model_trainer.decoder)
            model_trainer.attach(callback)

        return model_trainer
