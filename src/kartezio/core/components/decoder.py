import time
from abc import ABC
from typing import Dict, List

import numpy as np

from kartezio.core.components.adapter import AdapterMono
from kartezio.core.components.aggregation import Aggregation
from kartezio.core.components.base import Component, register
from kartezio.core.components.endpoint import Endpoint
from kartezio.core.components.genotype import Genotype
from kartezio.core.components.library import Library
from kartezio.core.population import Population


class Decoder(Component, ABC):
    def __init__(
        self, n_inputs: int, n_nodes: int, library: Library, endpoint: Endpoint = None
    ):
        super().__init__()
        if endpoint is None:
            n_outputs = 1
        else:
            n_outputs = len(endpoint.inputs)
        self.adapter = AdapterMono(
            n_inputs,
            n_nodes,
            n_outputs=n_outputs,
            n_parameters=library.max_parameters,
            n_connections=library.max_arity,
        )
        self.library = library
        self.endpoint = endpoint

    def decode(self, genotype: Genotype, x: List[np.ndarray]):
        all_y_pred = []
        all_times = []
        graphs = self.parse_to_graphs(genotype)

        # for each image
        for xi in x:
            start_time = time.time()
            y_pred = self._parse_one(genotype, graphs, xi)
            if self.endpoint is not None:
                y_pred = self.endpoint.call(y_pred)
            all_times.append(time.time() - start_time)
            all_y_pred.append(y_pred)
        whole_time = np.mean(np.array(all_times))
        return all_y_pred, whole_time

    def decode_population(self, population: Population, x: List[np.ndarray]) -> List:
        y_pred = []
        for i in range(1, population.size):
            y, t = self.decode(population.individuals[i], x)
            population.set_time(i, t)
            y_pred.append(y)
        return y_pred

    def __to_dict__(self) -> Dict:
        return {
            "genotype": {
                "inputs": self.infos.inputs,
                "nodes": self.infos.nodes,
                "outputs": self.infos.outputs,
                "parameters": self.infos.parameters,
                "connections": self.infos.connections,
            },
            "library": self.library.__to_dict__(),
            "endpoint": self.endpoint.__to_dict__(),
        }

    def _parse_one_graph(self, genotype: Genotype, graph_source):
        next_indices = graph_source.copy()
        output_tree = graph_source.copy()
        while next_indices:
            next_index = next_indices.pop()

            if next_index < self.adapter.n_inputs:
                continue
            function_index = self.adapter.read_function(
                genotype, next_index - self.adapter.n_inputs
            )
            active_connections = self.library.arity_of(function_index)
            next_connections = set(
                self.adapter.read_active_connections(
                    genotype, next_index - self.adapter.n_inputs, active_connections
                )
            )
            next_indices = next_indices.union(next_connections)
            output_tree = output_tree.union(next_connections)
        return sorted(list(output_tree))

    def parse_to_graphs(self, genotype: Genotype):
        outputs = self.adapter.read_outputs(genotype)
        graphs_list = [self._parse_one_graph(genotype, {output}) for output in outputs]
        return graphs_list

    def _x_to_output_map(self, genotype: Genotype, graphs_list: List, x: List):
        output_map = {i: x[i].copy() for i in range(self.adapter.n_inputs)}
        for graph in graphs_list:
            for node in graph:
                # inputs are already in the map
                if node < self.adapter.n_inputs:
                    continue
                node_index = node - self.adapter.n_inputs
                # fill the map with active nodes
                function_index = self.adapter.read_function(genotype, node_index)
                arity = self.library.arity_of(function_index)
                connections = self.adapter.read_active_connections(
                    genotype, node_index, arity
                )
                inputs = [output_map[c] for c in connections]
                p = self.adapter.read_parameters(genotype, node_index)
                value = self.library.execute(function_index, inputs, p)
                output_map[node] = value
        return output_map

    def _parse_one(self, genotype: Genotype, graphs_list: List, x: List):
        # fill output_map with inputs
        output_map = self._x_to_output_map(genotype, graphs_list, x)
        return [
            output_map[output_gene]
            for output_gene in self.adapter.read_outputs(genotype)
        ]


@register(Decoder, "sequential")
class SequentialDecoder(Decoder):
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Component":
        pass

    def to_iterative_decoder(self, aggregation):
        return DecoderIterative(self.infos, self.library, aggregation, self.endpoint)

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
        return SequentialDecoder(shape, library, endpoint)

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
