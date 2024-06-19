import time
from abc import ABC
from typing import Dict, List

import numpy as np

from kartezio.core.components.adapter import AdapterMono
from kartezio.core.components.base import Component, register
from kartezio.core.components.endpoint import Endpoint
from kartezio.core.components.genotype import Genotype
from kartezio.core.components.library import Library
from kartezio.core.components.reduction import Reduction
from kartezio.core.population import Population


class Decoder(Component, ABC):
    def __init__(
        self,
        n_inputs: int,
        n_nodes: int,
        library: Library,
        endpoint: Endpoint = None,
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

    def decode_population(
        self, population: Population, x: List[np.ndarray]
    ) -> List:
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
    
    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Decoder":
        n_inputs = dict_infos["metadata"]["n_in"]
        n_nodes = dict_infos["metadata"]["columns"]
        print(dict_infos["endpoint"])
        return SequentialDecoder(
            n_inputs,
            n_nodes,
            Library.__from_dict__(dict_infos),
            Endpoint.__from_dict__(dict_infos["endpoint"]),
        )

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
                    genotype,
                    next_index - self.adapter.n_inputs,
                    active_connections,
                )
            )
            next_indices = next_indices.union(next_connections)
            output_tree = output_tree.union(next_connections)
        return sorted(list(output_tree))

    def parse_to_graphs(self, genotype: Genotype):
        outputs = self.adapter.read_outputs(genotype)
        graphs_list = [
            self._parse_one_graph(genotype, {output}) for output in outputs
        ]
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
                function_index = self.adapter.read_function(
                    genotype, node_index
                )
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
    def to_iterative_decoder(self, reduction: Reduction) -> "IterativeDecoder":
        return IterativeDecoder(
            self.infos, self.library, reduction, self.endpoint
        )

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "SequentialDecoder":
        return super().__from_dict__(dict_infos)

    def __to_dict__(self) -> Dict:
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


@register(Decoder, "iterative")
class IterativeDecoder(Decoder):
    def __init__(self, n_inputs, n_nodes, library, endpoint, reduction):
        super().__init__(n_inputs, n_nodes, library, endpoint)
        self.reduction = reduction

    def decode(self, genotype: Genotype, x: List[np.ndarray]):
        all_y_pred = []
        all_times = []
        graphs = self.parse_to_graphs(genotype)
        for series in x:
            start_time = time.time()
            y_pred_series = []
            # for each image
            for xi in x:
                y_pred = self._parse_one(genotype, graphs, xi)
                y_pred_series.append(y_pred)
            y_pred = self.reduction.call(y_pred_series)
            if self.endpoint is not None:
                y_pred = self.endpoint.call(y_pred)
            all_times.append(time.time() - start_time)
            all_y_pred.append(y_pred)
        whole_time = np.mean(np.array(all_times))
        return all_y_pred, whole_time
