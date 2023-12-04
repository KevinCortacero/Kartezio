"""

"""
import ast
import copy
import random
import time
from abc import ABC
from builtins import print
from dataclasses import dataclass, field
from pprint import pprint
from typing import Callable, Dict, List, Sequence

import numpy as np
from tabulate import tabulate

from kartezio.model.helpers import Factory, Observer, Prototype, singleton
from kartezio.model.types import KType


class Components:
    __registry = {}

    @classmethod
    def register(cls, _class, component, name, replace: bool):
        assert isinstance(_class, type), f"{_class} is not a Class!"
        assert issubclass(_class, Component), f"{_class} is not a Component!"
        class_name = _class.__name__
        if class_name not in cls.__registry.keys():
            cls.__registry[class_name] = {}

        if name in cls.__registry[class_name].keys():
            if not replace:
                raise KeyError(
                    f"Error registering {class_name} called '{name}'."
                    f"Here is the list of all registered {class_name} components:"
                    f"{cls.__registry[class_name].keys()}"
                )
        cls.__registry[component][name] = _class

    @classmethod
    def display(cls):
        pprint(cls.__registry)

    @classmethod
    def instantiate(cls, class_name, name, *args, **kwargs):
        return cls.__registry[class_name][name](*args, **kwargs)


class BaseComponent(ABC):
    def __init__(self, name: str):
        self.name = name

    def to_toml(self):
        return {
            "name": self.name,
        }


class UpdatableComponent(BaseComponent, Observer, ABC):
    def __init__(self, name):
        super().__init__(name)


class Node(Component, ABC, component="Node"):
    def __init__(self, fn: Callable, **kwargs):
        assert callable(
            fn
        ), f"given 'function' {fn.__name__} is not callable! (type: {type(fn)})"
        super().__init__()
        self._fn = fn
        self.__kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class Preprocessing(Node, ABC, component="Preprocessing"):
    """
    Preprocessing node, called before training loop.
    """

    pass


class Primitive(Node, ABC, component="Primitive"):
    """
    Single graph node for the CGP Graph.
    """

    def __init__(self, fn: Callable, inputs, output, parameters):
        super().__init__(fn)
        self.inputs = inputs
        self.output = output
        self.parameters = parameters


class Endpoint(Node, component="Endpoint"):
    """
    Last node called to produce final outputs. Called in training loop,
    not submitted to evolution.
    """

    def __init__(self, name: str, fn: Callable, inputs, **kwargs):
        super().__init__(fn, **kwargs)
        self.inputs = inputs
        self._register(self.__class__, "Endpoint", name)

    def to_toml(self):
        return {
            "name": self.name,
        }


class Aggregation(Node, ABC, component="Aggregation"):
    def __init__(
        self,
        aggregation: Callable,
        inputs,
        post_aggregation: Callable = None,
    ):
        super().__init__(aggregation)
        self.post_aggregation = post_aggregation
        self.inputs = inputs

    def call(self, x: List):
        y = []
        for i in range(len(self.inputs)):
            if self.post_aggregation:
                y.append(self.post_aggregation(self._fn(x[i])))
            else:
                y.append(self._fn(x[i]))
        return y


class Library(BaseComponent):
    def __init__(self, output_type):
        super().__init__(output_type)
        self._primitives: Dict[int, Primitive] = {}
        self.output_type = output_type

    def to_toml(self):
        return {
            "output_type": self.output_type,
            "primitives": {
                str(i): self.name_of(i) for i in range(len(self._primitives))
            },
        }

    @staticmethod
    def from_json(json_data):
        library = Library()
        for node_name in json_data:
            library.add_primitive(node_name)
        return library

    def add_primitive(self, primitive: Primitive):
        self._primitives[len(self._primitives)] = primitive

    def add_library(self, library):
        for p in library.primitives:
            self.add_primitive(p)

    def name_of(self, i):
        return self._primitives[i].name

    def arity_of(self, i):
        return len(self._primitives[i].inputs)

    def parameters_of(self, i):
        return self._primitives[i].parameters

    def execute(self, f_index, x, args):
        return self._primitives[f_index](x, args)

    def display(self):
        headers = ["Id", "Name", "Inputs", "Outputs", "Arity", "Parameters"]
        full_list = []
        for i, primitive in self._primitives.items():
            one_primitive_infos = [
                i,
                self.name_of(i),
                primitive.inputs,
                primitive.output,
                self.arity_of(i),
                self.parameters_of(i),
            ]
            full_list.append(one_primitive_infos)
        table_name = f"  {self.output_type} Library  "
        print("─" * len(table_name))
        print(table_name)
        print(
            tabulate(
                full_list,
                tablefmt="simple_grid",
                headers=headers,
                numalign="center",
                stralign="center",
            )
        )

    @property
    def random_index(self):
        return random.choice(self.keys)

    @property
    def last_index(self):
        return len(self._primitives) - 1

    @property
    def primitives(self):
        return list(self._primitives.values())

    @property
    def keys(self):
        return list(self._primitives.keys())

    @property
    def max_arity(self):
        return max([self.arity_of(i) for i in self.keys])

    @property
    def max_parameters(self):
        return max([self.parameters_of(i) for i in self.keys])

    @property
    def size(self):
        return len(self._primitives)


class BaseGenotype(BaseComponent, Prototype, ABC):
    def clone(self):
        return copy.deepcopy(self)


@dataclass
class GenotypeInfos:
    n_inputs: int = 3
    n_nodes: int = 10
    n_outputs: int = 1
    n_connections: int = 2
    n_parameters: int = 2
    in_idx: int = field(init=False, repr=False)
    func_idx: int = field(init=False, repr=False)
    con_idx: int = field(init=False, repr=False)
    nodes_idx = None
    out_idx = None
    para_idx = None
    w: int = field(init=False)
    h: int = field(init=False)
    prototype: BaseGenotype = None

    def __post_init__(self):
        self.in_idx = 0
        self.func_idx = 0
        self.con_idx = 1
        self.nodes_idx = self.n_inputs
        self.out_idx = self.nodes_idx + self.n_nodes
        self.para_idx = self.con_idx + self.n_connections
        self.w = 1 + self.n_connections + self.n_parameters
        self.h = self.n_inputs + self.n_nodes + self.n_outputs
        self.prototype = KGenotypeArray(shape=(self.h, self.w))

    @staticmethod
    def from_json(json_data):
        return GenotypeInfos(
            json_data["inputs"],
            json_data["nodes"],
            json_data["outputs"],
            json_data["parameters"],
            json_data["connections"],
        )

    def new(self):
        return self.prototype.clone()


class KGenotypeArray(BaseGenotype):
    """
    Only store "DNA" into a Numpy array (ndarray)
    No metadata stored in DNA to avoid duplicates
    Avoiding RAM overload: https://refactoring.guru/design-patterns/flyweight
    Default genotype would be: 3 inputs, 10 function nodes and 1 output (M=14),
    with 1 function, 2 connections and 2 parameters (N=5), giving final 2D shape (14, 5).
    """

    def __init__(self, shape: tuple = (14, 5)):
        super().__init__("Genotype")
        self.__genes = np.zeros(shape=shape, dtype=np.uint8)

    def __copy__(self):
        new = self.__class__(*self.__genes.shape)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo={}):
        new = self.__class__(self.__genes.shape)
        new.__genes = self.__genes.copy()
        return new

    def __getitem__(self, item):
        return self.__genes.__getitem__(item)

    def __setitem__(self, key, value):
        return self.__genes.__setitem__(key, value)

    @classmethod
    def from_ndarray(cls, genes: np.ndarray):
        genotype = KGenotypeArray()
        genotype.__genes = genes
        return genotype

    @classmethod
    def from_json(cls, json_data):
        genes = np.asarray(ast.literal_eval(json_data["sequence"]))
        return cls.from_ndarray(genes)

    @property
    def sequence(self):
        return self.__genes


class GenomeFactory(Factory):
    def __init__(self, prototype: BaseGenotype):
        super().__init__(prototype)


@dataclass
class GenotypeAdapter(ABC):
    """
    Adpater Design Pattern: https://refactoring.guru/design-patterns/adapter
    """

    infos: GenotypeInfos


class GenotypeWriter(GenotypeAdapter):
    def write_function(self, genome, node, function_id):
        genome[self.infos.nodes_idx + node, self.infos.func_idx] = function_id

    def write_connections(self, genome, node, connections):
        genome[
            self.infos.nodes_idx + node, self.infos.con_idx : self.infos.para_idx
        ] = connections

    def write_parameters(self, genome, node, parameters):
        genome[self.infos.nodes_idx + node, self.infos.para_idx :] = parameters

    def write_output_connection(self, genome, output_index, connection):
        genome[self.infos.out_idx + output_index, self.infos.con_idx] = connection


class GenotypeReader(GenotypeAdapter):
    def read_function(self, genome, node):
        return genome[self.infos.nodes_idx + node, self.infos.func_idx]

    def read_connections(self, genome, node):
        return genome[
            self.infos.nodes_idx + node, self.infos.con_idx : self.infos.para_idx
        ]

    def read_active_connections(self, genome, node, active_connections):
        return genome[
            self.infos.nodes_idx + node,
            self.infos.con_idx : self.infos.con_idx + active_connections,
        ]

    def read_parameters(self, genome, node):
        return genome[self.infos.nodes_idx + node, self.infos.para_idx :]

    def read_outputs(self, genome):
        return genome[self.infos.out_idx :, :]


class GenomeReaderWriter(GenotypeReader, GenotypeWriter):
    pass


class Decoder(GenotypeReader, ABC):
    def decode_population(self, population, x: List) -> List:
        y_pred = []
        for i in range(1, population.size):
            y, t = self.decode(population.individuals[i], x)
            population.set_time(i, t)
            y_pred.append(y)
        return y_pred

    def to_toml(self) -> Dict:
        return {
            "genotype": {
                "inputs": self.infos.inputs,
                "nodes": self.infos.nodes,
                "outputs": self.infos.outputs,
                "parameters": self.infos.parameters,
                "connections": self.infos.connections,
            },
            "library": self.library.to_toml(),
            "endpoint": self.endpoint.to_toml(),
            "mode": "sequential",
        }


class DecoderSequential(Decoder):
    def __init__(
        self, n_inputs: int, n_nodes: int, library: Library, endpoint: Endpoint = None
    ):
        if endpoint is None:
            n_outputs = 1
        else:
            n_outputs = len(endpoint.inputs)
        infos = GenotypeInfos(
            n_inputs,
            n_nodes,
            n_outputs=n_outputs,
            n_parameters=library.max_parameters,
            n_connections=library.max_arity,
        )
        super().__init__(infos)
        self.library = library
        self.endpoint = endpoint

    def to_iterative_decoder(self, stacker):
        return DecoderIterative(self.infos, self.library, stacker, self.endpoint)

    def to_toml(self) -> Dict:
        return {
            "genotype": {
                "inputs": self.infos.n_inputs,
                "nodes": self.infos.n_nodes,
                "outputs": self.infos.n_outputs,
                "parameters": self.infos.n_parameters,
                "connections": self.infos.n_connections,
            },
            "library": self.library.to_toml(),
            "endpoint": self.endpoint.to_toml(),
            "mode": "sequential",
        }

    @staticmethod
    def from_json(json_data):
        shape = GenotypeInfos.from_json(json_data["genotype"])
        library = None  # KLibrary.from_json(json_data["functions"])
        endpoint = None  # KEndpoint.from_json(json_data["endpoint"])
        if json_data["mode"] == "series":
            stacker = Aggregation.from_json(json_data["stacker"])
            return DecoderIterative(shape, library, stacker, endpoint)
        return DecoderSequential(shape, library, endpoint)

    def _parse_one_graph(self, genome, graph_source):
        next_indices = graph_source.copy()
        output_tree = graph_source.copy()
        while next_indices:
            next_index = next_indices.pop()
            if next_index < self.infos.n_inputs:
                continue
            function_index = self.read_function(
                genome, next_index - self.infos.n_inputs
            )
            active_connections = self.library.arity_of(function_index)
            next_connections = set(
                self.read_active_connections(
                    genome, next_index - self.infos.n_inputs, active_connections
                )
            )
            next_indices = next_indices.union(next_connections)
            output_tree = output_tree.union(next_connections)
        return sorted(list(output_tree))

    def parse_to_graphs(self, genome):
        outputs = self.read_outputs(genome)
        graphs_list = [
            self._parse_one_graph(genome, {output[self.infos.con_idx]})
            for output in outputs
        ]
        return graphs_list

    def _x_to_output_map(self, genome: BaseGenotype, graphs_list: List, x: List):
        output_map = {i: x[i].copy() for i in range(self.infos.n_inputs)}
        for graph in graphs_list:
            for node in graph:
                # inputs are already in the map
                if node < self.infos.n_inputs:
                    continue
                node_index = node - self.infos.n_inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                arity = self.library.arity_of(function_index)
                connections = self.read_active_connections(genome, node_index, arity)
                inputs = [output_map[c] for c in connections]
                p = self.read_parameters(genome, node_index)
                value = self.library.execute(function_index, inputs, p)
                output_map[node] = value
        return output_map

    def _parse_one(self, genome: BaseGenotype, graphs_list: List, x: List):
        # fill output_map with inputs
        output_map = self._x_to_output_map(genome, graphs_list, x)
        return [
            output_map[output_gene[self.infos.con_idx]]
            for output_gene in self.read_outputs(genome)
        ]

    def active_size(self, genome):
        node_list = []
        graphs_list = self.parse_to_graphs(genome)
        for graph in graphs_list:
            for node in graph:
                if node < self.infos.inputs:
                    continue
                if node < self.infos.out_idx:
                    node_list.append(node)
                else:
                    continue
        return len(node_list)

    def node_histogram(self, genome):
        nodes = {}
        graphs_list = self.parse_to_graphs(genome)
        for graph in graphs_list:
            for node in graph:
                # inputs are already in the map
                if node < self.infos.inputs:
                    continue
                node_index = node - self.infos.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                function_name = self.library.name_of(function_index)
                if function_name not in nodes.keys():
                    nodes[function_name] = 0
                nodes[function_name] += 1
        return nodes

    def get_last_node(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        output_functions = []
        for graph in graphs_list:
            for node in graph[-1:]:
                # inputs are already in the map
                if node < self.infos.inputs:
                    print(f"output {node} directly connected to input.")
                    continue
                node_index = node - self.infos.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                function_name = self.library.name_of(function_index)
                output_functions.append(function_name)
        return output_functions

    def get_first_node(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        input_functions = []

        for graph in graphs_list:
            for node in graph:
                if node < self.infos.inputs:
                    print(f"output {node} directly connected to input.")
                    continue
                node_index = node - self.infos.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                function_name = self.library.name_of(function_index)
                arity = self.library.arity_of(function_index)
                connections = self.read_active_connections(genome, node_index, arity)
                for c in connections:
                    if c < self.infos.inputs:
                        input_functions.append(function_name)
        return input_functions

    def bigrams(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        outputs = self.read_outputs(genome)
        print(graphs_list)
        bigram_list = []
        for i, graph in enumerate(graphs_list):
            for j, node in enumerate(graph):
                if node < self.infos.inputs:
                    continue
                node_index = node - self.infos.inputs
                function_index = self.read_function(genome, node_index)
                fname = self.library.name_of(function_index)
                arity = self.library.arity_of(function_index)
                connections = self.read_active_connections(genome, node_index, arity)
                for k, c in enumerate(connections):
                    if c < self.infos.inputs:
                        in_name = f"IN-{c}"
                        pair = (f"{fname}", in_name)
                        """
                        if arity == 1:
                            pair = (f"{fname}", in_name)
                        else:
                            pair = (f"{fname}-{k}", in_name)
                        """

                    else:
                        f2_index = self.read_function(genome, c - self.infos.inputs)
                        f2_name = self.library.name_of(f2_index)
                        """
                        if arity == 1:
                            pair = (f"{fname}", f2_name)
                        else:
                            pair = (f"{fname}-{k}", f2_name)
                        """
                        pair = (f"{fname}", f2_name)
                    bigram_list.append(pair)

            f_last = self.read_function(genome, outputs[i][1] - self.infos.inputs)
            fname = self.library.name_of(f_last)
            pair = (f"OUT-{i}", fname)
            bigram_list.append(pair)
        print(bigram_list)
        return bigram_list

    def function_distribution(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        active_list = []
        for graph in graphs_list:
            for node in graph:
                if node < self.infos.inputs:
                    continue
                if node >= self.infos.out_idx:
                    continue
                active_list.append(node)
        functions = []
        is_active = []
        for i, _ in enumerate(genome.genes):
            if i < self.infos.inputs:
                continue
            if i >= self.infos.out_idx:
                continue
            node_index = i - self.infos.inputs
            function_index = self.read_function(genome, node_index)
            function_name = self.library.name_of(function_index)
            functions.append(function_name)
            is_active.append(i in active_list)
        return functions, is_active

    def decode(self, genome, x):
        """Decode the Genome given a list of inputs

        Args:
            genome (BaseGenotype): [description]
            x (List): [description]

        Returns:
            [type]: [description]
        """
        all_y_pred = []
        all_times = []
        graphs = self.parse_to_graphs(genome)

        # for each image
        for xi in x:
            start_time = time.time()
            y_pred = self._parse_one(genome, graphs, xi)
            if self.endpoint is not None:
                y_pred = self.endpoint(y_pred)
            all_times.append(time.time() - start_time)
            all_y_pred.append(y_pred)
        whole_time = np.mean(np.array(all_times))
        return all_y_pred, whole_time


class DecoderIterative(Decoder):
    def __init__(self, shape, bundle, stacker, endpoint):
        super().__init__(shape, bundle, endpoint)
        self.stacker = stacker

    def parse(self, genome, x):
        """Decode the Genome given a list of inputs
        Args:
            genome (BaseGenotype): [description]
            x (List): [description]
        Returns:
            [type]: [description]
        """
        all_y_pred = []
        all_times = []
        graphs = self.parse_to_graphs(genome)
        for series in x:
            start_time = time.time()
            y_pred_series = []
            # for each image

            for xi in series:
                y_pred = self._parse_one(genome, graphs, xi)
                y_pred_series.append(y_pred)

            y_pred = self.endpoint.call(self.stacker.call(y_pred_series))

            all_times.append(time.time() - start_time)
            all_y_pred.append(y_pred)

        whole_time = np.mean(np.array(all_times))
        return all_y_pred, whole_time

    def dumps(self) -> dict:
        json_data = super().dumps()
        json_data["mode"] = "series"
        json_data["stacker"] = self.stacker.dumps()
        return json_data


class KartezioToCode(Decoder):
    def to_python_class(self, node_name, genome):
        pass
