from dataclasses import InitVar, dataclass, field

from kartezio.model.base import ModelCGP
from kartezio.model.components import (
    GenomeFactory,
    GenomeShape,
    KartezioBundle,
    KartezioEndpoint,
    KartezioParser,
    KartezioStacker,
    ParserChain,
)
from kartezio.model.evolution import KartezioFitness, KartezioMutation
from kartezio.model.registry import registry
from kartezio.mutation import GoldmanWrapper, MutationAllRandom
from kartezio.stacker import StackerMean
from kartezio.strategy import OnePlusLambda


@dataclass
class ModelContext:
    genome_shape: GenomeShape = field(init=False)
    genome_factory: GenomeFactory = field(init=False)
    instance_method: KartezioMutation = field(init=False)
    mutation_method: KartezioMutation = field(init=False)
    fitness: KartezioFitness = field(init=False)
    stacker: KartezioStacker = field(init=False)
    endpoint: KartezioEndpoint = field(init=False)
    bundle: KartezioBundle = field(init=False)
    parser: KartezioParser = field(init=False)
    series_mode: bool = field(init=False)
    inputs: InitVar[int] = 3
    nodes: InitVar[int] = 10
    outputs: InitVar[int] = 1
    arity: InitVar[int] = 2
    parameters: InitVar[int] = 2

    def __post_init__(
        self, inputs: int, nodes: int, outputs: int, arity: int, parameters: int
    ):
        self.genome_shape = GenomeShape(inputs, nodes, outputs, arity, parameters)
        self.genome_factory = GenomeFactory(self.genome_shape.prototype)

    def set_bundle(self, bundle: KartezioBundle):
        self.bundle = bundle

    def set_endpoint(self, endpoint: KartezioEndpoint):
        self.endpoint = endpoint

    def set_instance_method(self, instance_method):
        self.instance_method = instance_method

    def set_mutation_method(self, mutation_method):
        self.mutation_method = mutation_method

    def set_fitness(self, fitness):
        self.fitness = fitness

    def compile_parser(self, series_mode, series_stacker):
        if series_mode:
            if type(series_stacker) == str:
                series_stacker = registry.stackers.instantiate(series_stacker)
            parser = ParserChain(
                self.genome_shape, self.bundle, series_stacker, self.endpoint
            )
        else:
            parser = KartezioParser(self.genome_shape, self.bundle, self.endpoint)
        self.parser = parser
        self.series_mode = series_mode


class ModelBuilder:
    def __init__(self):
        self.__context = None

    def create(
        self,
        endpoint,
        bundle,
        inputs=3,
        nodes=10,
        outputs=1,
        arity=2,
        parameters=2,
        series_mode=False,
        series_stacker=StackerMean(),
    ):
        self.__context = ModelContext(inputs, nodes, outputs, arity, parameters)
        self.__context.set_endpoint(endpoint)
        self.__context.set_bundle(bundle)
        self.__context.compile_parser(series_mode, series_stacker)

    def set_instance_method(self, instance_method):
        if type(instance_method) == str:
            if instance_method == "random":
                shape = self.__context.genome_shape
                n_nodes = self.__context.bundle.size
                instance_method = MutationAllRandom(shape, n_nodes)
        self.__context.set_instance_method(instance_method)

    def set_mutation_method(
        self, mutation, node_mutation_rate, output_mutation_rate, use_goldman=True
    ):
        if type(mutation) == str:
            shape = self.__context.genome_shape
            n_nodes = self.__context.bundle.size
            mutation = registry.mutations.instantiate(
                mutation, shape, n_nodes, node_mutation_rate, output_mutation_rate
            )
        if use_goldman:
            parser = self.__context.parser
            mutation = GoldmanWrapper(mutation, parser)
        self.__context.set_mutation_method(mutation)

    def set_fitness(self, fitness):
        if type(fitness) == str:
            fitness = registry.fitness.instantiate(fitness)
        self.__context.set_fitness(fitness)

    def compile(self, generations, _lambda, callbacks=None, dataset_inputs=None):
        factory = self.__context.genome_factory
        instance_method = self.__context.instance_method
        mutation_method = self.__context.mutation_method
        fitness = self.__context.fitness
        parser = self.__context.parser

        if parser.endpoint.arity != parser.shape.outputs:
            raise ValueError(
                f"Endpoint [{parser.endpoint.name}] requires {parser.endpoint.arity} output nodes. ({parser.shape.outputs} given)"
            )

        if self.__context.series_mode:
            if not isinstance(parser.stacker, KartezioStacker):
                raise ValueError(f"Stacker {parser.stacker} has not been properly set.")

            if parser.stacker.arity != parser.shape.outputs:
                raise ValueError(
                    f"Stacker [{parser.stacker.name}] requires {parser.stacker.arity} output nodes. ({parser.shape.outputs} given)"
                )

        if not isinstance(fitness, KartezioFitness):
            raise ValueError(f"Fitness {fitness} has not been properly set.")

        if not isinstance(mutation_method, KartezioMutation):
            raise ValueError(f"Mutation {mutation_method} has not been properly set.")

        if dataset_inputs and (dataset_inputs != parser.shape.inputs):
            raise ValueError(
                f"Model has {parser.shape.inputs} input nodes. ({dataset_inputs} given by the dataset)"
            )

        strategy = OnePlusLambda(
            _lambda, factory, instance_method, mutation_method, fitness
        )
        model = ModelCGP(generations, strategy, parser)
        if callbacks:
            for callback in callbacks:
                callback.set_parser(parser)
                model.attach(callback)
        return model
