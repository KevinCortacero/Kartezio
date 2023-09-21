"""

"""
import ast
import copy
import random
import time
from abc import ABC, abstractmethod
from builtins import print
from dataclasses import dataclass, field
from typing import List

import numpy as np
from numena.io.json import Serializable

from kartezio.model.helpers import Factory, Observer, Prototype
from kartezio.model.registry import registry


class KartezioComponent(Serializable, ABC):
    pass


class KartezioNode(KartezioComponent, ABC):
    """
    Single graph node for the Cartesian Graph.
    One node can be a simple function (e.g. Threshold, Subtract...), but also a more complex function such as an KartezioEndpoint.
    """

    def __init__(self, name: str, symbol: str, arity: int, args: int, sources=None):
        """
        Args:
            name (str): Name of the node
            symbol (str): Abbreviation of the node, it must be written in capital letters with 3 or 4 characters (e.g. "ADD", "NOT", "OPEN"..)
            arity (int): Number of inputs the node needs (e.g. 2 for addition (x1+x2), 1 for sqrt (sqrt(x1)))
            args (int): Number of parameters the node needs (e.g. 0 for addition (x1+x2), 1 for threshold (threshold(x1, p1)))
        >>> threshold_node = Threshold("threshold", "TRSH", 1, 1)
        >>> watershed_endpoint = Watershed("watershed", "WSHD", 2, 0)
        """
        self.name = name
        self.symbol = symbol
        self.arity = arity
        self.args = args
        self.sources = sources

    @abstractmethod
    def call(self, x: List, args: List = None):
        pass

    def dumps(self) -> dict:
        return {
            "name": self.name,
            "abbv": self.symbol,
            "arity": self.arity,
            "args": self.args,
            "kwargs": self._to_json_kwargs(),
        }

    @abstractmethod
    def _to_json_kwargs(self) -> dict:
        pass


class KartezioEndpoint(KartezioNode, ABC):
    """
    Terminal KartezioNode, executed after graph parsing.
    Not submitted to evolution.
    """

    def __init__(self, name: str, symbol: str, arity: int, outputs_keys: list):
        super().__init__(name, symbol, arity, 0)
        self.outputs_keys = outputs_keys

    @staticmethod
    def from_json(json_data):
        return registry.endpoints.instantiate(json_data["abbv"], **json_data["kwargs"])


class KartezioPreprocessing(KartezioNode, ABC):
    """
    First KartezioNode, executed before evolution loop.
    Not submitted to evolution.
    """

    def __init__(self, name: str, symbol: str):
        super().__init__(name, symbol, 1, 0)


class KartezioGenome(KartezioComponent, Prototype):
    """
    Only store "DNA" in a numpy array
    No metadata stored in DNA to avoid duplicates
    Avoiding RAM overload: https://refactoring.guru/design-patterns/flyweight
    Default genome would be: 3 inputs, 10 function nodes (2 connections and 2 parameters), 1 output,
    so with shape (14, 5)

    Args:
        Prototype ([type]): [description]

    Returns:
        [type]: [description]
    """

    def dumps(self) -> dict:
        pass

    def __init__(self, shape: tuple = (14, 5), sequence: np.ndarray = None):
        if sequence is not None:
            self.sequence = sequence
        else:
            self.sequence = np.zeros(shape=shape, dtype=np.uint8)

    def __copy__(self):
        new = self.__class__(*self.sequence.shape)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo={}):
        new = self.__class__(*self.sequence.shape)
        new.sequence = self.sequence.copy()
        return new

    def __getitem__(self, item):
        return self.sequence.__getitem__(item)

    def __setitem__(self, key, value):
        return self.sequence.__setitem__(key, value)

    def clone(self):
        return copy.deepcopy(self)

    @staticmethod
    def from_json(json_data):
        sequence = np.asarray(ast.literal_eval(json_data["sequence"]))
        return KartezioGenome(sequence=sequence)


class GenomeFactory(Factory):
    def __init__(self, prototype: KartezioGenome):
        super().__init__(prototype)


class GenomeAdapter(KartezioComponent, ABC):
    """
    Adpater Design Pattern: https://refactoring.guru/design-patterns/adapter
    """

    def __init__(self, shape):
        self.shape = shape


class GenomeWriter(GenomeAdapter):
    def write_function(self, genome, node, function_id):
        genome[self.shape.nodes_idx + node, self.shape.func_idx] = function_id

    def write_connections(self, genome, node, connections):
        genome[
            self.shape.nodes_idx + node, self.shape.con_idx : self.shape.para_idx
        ] = connections

    def write_parameters(self, genome, node, parameters):
        genome[self.shape.nodes_idx + node, self.shape.para_idx :] = parameters

    def write_output_connection(self, genome, output_index, connection):
        genome[self.shape.out_idx + output_index, self.shape.con_idx] = connection


class GenomeReader(GenomeAdapter):
    def read_function(self, genome, node):
        return genome[self.shape.nodes_idx + node, self.shape.func_idx]

    def read_connections(self, genome, node):
        return genome[
            self.shape.nodes_idx + node, self.shape.con_idx : self.shape.para_idx
        ]

    def read_active_connections(self, genome, node, active_connections):
        return genome[
            self.shape.nodes_idx + node,
            self.shape.con_idx : self.shape.con_idx + active_connections,
        ]

    def read_parameters(self, genome, node):
        return genome[self.shape.nodes_idx + node, self.shape.para_idx :]

    def read_outputs(self, genome):
        return genome[self.shape.out_idx :, :]


class GenomeReaderWriter(GenomeReader, GenomeWriter):
    pass


@dataclass
class GenomeShape:
    inputs: int = 3
    nodes: int = 10
    outputs: int = 1
    connections: int = 2
    parameters: int = 2
    in_idx: int = field(init=False, repr=False)
    func_idx: int = field(init=False, repr=False)
    con_idx: int = field(init=False, repr=False)
    nodes_idx = None
    out_idx = None
    para_idx = None
    w: int = field(init=False)
    h: int = field(init=False)
    prototype = None

    def __post_init__(self):
        self.in_idx = 0
        self.func_idx = 0
        self.con_idx = 1
        self.nodes_idx = self.inputs
        self.out_idx = self.nodes_idx + self.nodes
        self.para_idx = self.con_idx + self.connections
        self.w = 1 + self.connections + self.parameters
        self.h = self.inputs + self.nodes + self.outputs
        self.prototype = KartezioGenome(shape=(self.h, self.w))

    @staticmethod
    def from_json(json_data):
        return GenomeShape(
            json_data["n_in"],
            json_data["columns"],
            json_data["n_out"],
            json_data["n_conn"],
            json_data["n_para"],
        )


class KartezioParser(GenomeReader):
    def __init__(self, shape, function_bundle, endpoint):
        super().__init__(shape)
        self.function_bundle = function_bundle
        self.endpoint = endpoint

    def to_series_parser(self, stacker):
        return ParserChain(self.shape, self.function_bundle, stacker, self.endpoint)

    def dumps(self) -> dict:
        return {
            "metadata": {
                "rows": 1,  # single row CGP
                "columns": self.shape.nodes,
                "n_in": self.shape.inputs,
                "n_out": self.shape.outputs,
                "n_para": self.shape.parameters,
                "n_conn": self.shape.connections,
            },
            "functions": self.function_bundle.ordered_list,
            "endpoint": self.endpoint.dumps(),
            "mode": "default",
        }

    @staticmethod
    def from_json(json_data):
        shape = GenomeShape.from_json(json_data["metadata"])
        bundle = KartezioBundle.from_json(json_data["functions"])
        endpoint = KartezioEndpoint.from_json(json_data["endpoint"])
        if json_data["mode"] == "series":
            stacker = KartezioStacker.from_json(json_data["stacker"])
            return ParserChain(shape, bundle, stacker, endpoint)
        return KartezioParser(shape, bundle, endpoint)

    def _parse_one_graph(self, genome, graph_source):
        next_indices = graph_source.copy()
        output_tree = graph_source.copy()
        while next_indices:
            next_index = next_indices.pop()
            if next_index < self.shape.inputs:
                continue
            function_index = self.read_function(genome, next_index - self.shape.inputs)
            active_connections = self.function_bundle.arity_of(function_index)
            next_connections = set(
                self.read_active_connections(
                    genome, next_index - self.shape.inputs, active_connections
                )
            )
            next_indices = next_indices.union(next_connections)
            output_tree = output_tree.union(next_connections)
        return sorted(list(output_tree))

    def parse_to_graphs(self, genome):
        outputs = self.read_outputs(genome)
        graphs_list = [
            self._parse_one_graph(genome, {output[self.shape.con_idx]})
            for output in outputs
        ]
        return graphs_list

    def _x_to_output_map(self, genome: KartezioGenome, graphs_list: List, x: List):
        output_map = {i: x[i].copy() for i in range(self.shape.inputs)}
        for graph in graphs_list:
            for node in graph:
                # inputs are already in the map
                if node < self.shape.inputs:
                    continue
                node_index = node - self.shape.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                arity = self.function_bundle.arity_of(function_index)
                connections = self.read_active_connections(genome, node_index, arity)
                inputs = [output_map[c] for c in connections]
                p = self.read_parameters(genome, node_index)
                value = self.function_bundle.execute(function_index, inputs, p)

                output_map[node] = value
        return output_map

    def _parse_one(self, genome: KartezioGenome, graphs_list: List, x: List):
        # fill output_map with inputs
        output_map = self._x_to_output_map(genome, graphs_list, x)
        return [
            output_map[output_gene[self.shape.con_idx]]
            for output_gene in self.read_outputs(genome)
        ]

    def active_size(self, genome):
        node_list = []
        graphs_list = self.parse_to_graphs(genome)
        for graph in graphs_list:
            for node in graph:
                if node < self.shape.inputs:
                    continue
                if node < self.shape.out_idx:
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
                if node < self.shape.inputs:
                    continue
                node_index = node - self.shape.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                function_name = self.function_bundle.symbol_of(function_index)
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
                if node < self.shape.inputs:
                    print(f"output {node} directly connected to input.")
                    continue
                node_index = node - self.shape.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                function_name = self.function_bundle.symbol_of(function_index)
                output_functions.append(function_name)
        return output_functions

    def get_first_node(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        input_functions = []

        for graph in graphs_list:
            for node in graph:
                if node < self.shape.inputs:
                    print(f"output {node} directly connected to input.")
                    continue
                node_index = node - self.shape.inputs
                # fill the map with active nodes
                function_index = self.read_function(genome, node_index)
                function_name = self.function_bundle.symbol_of(function_index)
                arity = self.function_bundle.arity_of(function_index)
                connections = self.read_active_connections(genome, node_index, arity)
                for c in connections:
                    if c < self.shape.inputs:
                        input_functions.append(function_name)
        return input_functions

    def bigrams(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        outputs = self.read_outputs(genome)
        print(graphs_list)
        bigram_list = []
        for i, graph in enumerate(graphs_list):
            for j, node in enumerate(graph):
                if node < self.shape.inputs:
                    continue
                node_index = node - self.shape.inputs
                function_index = self.read_function(genome, node_index)
                fname = self.function_bundle.symbol_of(function_index)
                arity = self.function_bundle.arity_of(function_index)
                connections = self.read_active_connections(genome, node_index, arity)
                for k, c in enumerate(connections):
                    if c < self.shape.inputs:
                        in_name = f"IN-{c}"
                        pair = (f"{fname}", in_name)
                        """
                        if arity == 1:
                            pair = (f"{fname}", in_name)
                        else:
                            pair = (f"{fname}-{k}", in_name)
                        """

                    else:
                        f2_index = self.read_function(genome, c - self.shape.inputs)
                        f2_name = self.function_bundle.symbol_of(f2_index)
                        """
                        if arity == 1:
                            pair = (f"{fname}", f2_name)
                        else:
                            pair = (f"{fname}-{k}", f2_name)
                        """
                        pair = (f"{fname}", f2_name)
                    bigram_list.append(pair)

            f_last = self.read_function(genome, outputs[i][1] - self.shape.inputs)
            fname = self.function_bundle.symbol_of(f_last)
            pair = (f"OUT-{i}", fname)
            bigram_list.append(pair)
        print(bigram_list)
        return bigram_list

    def function_distribution(self, genome):
        graphs_list = self.parse_to_graphs(genome)
        active_list = []
        for graph in graphs_list:
            for node in graph:
                if node < self.shape.inputs:
                    continue
                if node >= self.shape.out_idx:
                    continue
                active_list.append(node)
        functions = []
        is_active = []
        for i, _ in enumerate(genome.sequence):
            if i < self.shape.inputs:
                continue
            if i >= self.shape.out_idx:
                continue
            node_index = i - self.shape.inputs
            function_index = self.read_function(genome, node_index)
            function_name = self.function_bundle.symbol_of(function_index)
            functions.append(function_name)
            is_active.append(i in active_list)
        return functions, is_active

    def parse_population(self, population, x):
        y_pred = []
        for i in range(len(population.individuals)):
            y, t = self.parse(population.individuals[i], x)
            population.set_time(i, t)
            y_pred.append(y)
        return y_pred

    def parse(self, genome, x):
        """Decode the Genome given a list of inputs

        Args:
            genome (KartezioGenome): [description]
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
                y_pred = self.endpoint.call(y_pred)
            all_times.append(time.time() - start_time)
            all_y_pred.append(y_pred)
        whole_time = np.mean(np.array(all_times))
        return all_y_pred, whole_time


class ParserSequential(KartezioParser):
    """TODO: default Parser, KartezioParser becomes ABC"""

    pass


class ParserChain(KartezioParser):
    def __init__(self, shape, bundle, stacker, endpoint):
        super().__init__(shape, bundle, endpoint)
        self.stacker = stacker

    def parse(self, genome, x):
        """Decode the Genome given a list of inputs
        Args:
            genome (KartezioGenome): [description]
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


class KartezioToCode(KartezioParser):
    def to_python_class(self, node_name, genome):
        pass


class KartezioStacker(KartezioNode, ABC):
    def __init__(self, name: str, symbol: str, arity: int):
        super().__init__(name, symbol, arity, 0)

    def call(self, x: List, args: List = None):
        y = []
        for i in range(self.arity):
            Y = [xi[i] for xi in x]
            y.append(self.post_stack(self.stack(Y), i))
        return y

    @abstractmethod
    def stack(self, Y: List):
        pass

    def post_stack(self, x, output_index):
        return x

    @staticmethod
    def from_json(json_data):
        return registry.stackers.instantiate(
            json_data["abbv"], arity=json_data["arity"], **json_data["kwargs"]
        )


class ExportableNode(KartezioNode, ABC):
    def _to_json_kwargs(self) -> dict:
        return {}

    @abstractmethod
    def to_python(self, input_nodes: List, p: List, node_name: str):
        """

        Parameters
        ----------
        input_nodes :
        p :
        node_name :
        """
        pass

    @abstractmethod
    def to_cpp(self, input_nodes: List, p: List, node_name: str):
        """

        :param input_nodes:
        :type input_nodes:
        :param p:
        :type p:
        :param node_name:
        :type node_name:
        """
        pass


class KartezioCallback(KartezioComponent, Observer, ABC):
    def __init__(self, frequency=1):
        self.frequency = frequency
        self.parser = None

    def set_parser(self, parser):
        self.parser = parser

    def update(self, event):
        if event["n"] % self.frequency == 0 or event["force"]:
            self._callback(event["n"], event["name"], event["content"])

    def dumps(self) -> dict:
        return {}

    @abstractmethod
    def _callback(self, n, e_name, e_content):
        pass
