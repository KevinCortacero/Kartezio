import time
from abc import ABC
from typing import Dict, List

import numpy as np

from kartezio.core.components import (
    Endpoint,
    Genotype,
    KartezioComponent,
    Library,
    dump_component,
    fundamental,
    register,
)
from kartezio.evolution.population import Population


@fundamental()
class Adapter(KartezioComponent):
    """
    Adpater Design Pattern: https://refactoring.guru/design-patterns/adapter
    """

    FUNCTION = 0

    class ChromosomeInfo:
        def __init__(self, n_nodes, n_edges, n_parameters, n_functions):
            self.n_edges = n_edges
            self.n_parameters = n_parameters
            self.para_idx = 1 + n_edges
            self.shape = (n_nodes, 1 + self.n_edges + self.n_parameters)
            self.n_functions = n_functions

    def __init__(self, n_inputs, n_nodes, returns, libraries: List[Library]):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.returns = returns
        self.n_outputs = len(self.returns)
        self.out_idx = self.n_inputs + self.n_nodes
        self.chromosomes_infos = {
            library.rtype: Adapter.ChromosomeInfo(
                n_nodes,
                library.max_arity,
                library.max_parameters,
                library.size,
            )
            for library in libraries
        }
        self.types_map = {
            t: i for i, t in enumerate(self.chromosomes_infos.keys())
        }
        self.prototype = self.create_prototype()

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Adapter":
        pass

    def __to_dict__(self) -> Dict:
        return {
            "n_inputs": self.n_inputs,
            "n_nodes": self.n_nodes,
            "returns": self.returns,
        }

    def new_genotype(self):
        return self.prototype.clone()

    def create_prototype(self):
        genotype = Genotype(self.n_outputs)
        for chromosome, info in self.chromosomes_infos.items():
            genotype[chromosome] = np.zeros(info.shape, dtype=np.uint8)
        return genotype

    def set_function(
        self,
        genotype: Genotype,
        chromosome: str,
        node: int,
        function_id: int,
    ):
        genotype[chromosome][node, 0] = function_id

    def set_edges(
        self,
        genotype: Genotype,
        chromosome: str,
        node: int,
        edges,
    ):
        genotype[chromosome][
            node, 1 : self.chromosomes_infos[chromosome].para_idx
        ] = edges

    def set_parameters(
        self,
        genotype: Genotype,
        chromosome: str,
        node: int,
        parameters,
    ):
        genotype[chromosome][
            node, self.chromosomes_infos[chromosome].para_idx :
        ] = parameters

    def set_output(self, genotype: Genotype, output_index, connection):
        genotype["outputs"][output_index] = connection

    def get_function(self, genotype: Genotype, chromosome: str, node: int):
        return genotype[chromosome][node, 0]

    def get_edges(self, genotype: Genotype, chromosome: str, node: int):
        return genotype[chromosome][
            node, 1 : self.chromosomes_infos[chromosome].para_idx
        ]

    def get_active_edges(
        self,
        genotype: Genotype,
        chromosome: str,
        node: int,
        n_edges: int,
    ):
        return genotype[chromosome][node, 1 : 1 + n_edges]

    def get_parameters(self, genotype: Genotype, chromosome: str, node: int):
        return genotype[chromosome][
            node, self.chromosomes_infos[chromosome].para_idx :
        ]

    def get_outputs(self, genotype: Genotype):
        return genotype["outputs"]

    def to_chromosome_indices(self, types):
        return [self.types_map[_type] for _type in types]


@fundamental()
class Decoder(KartezioComponent, ABC):
    pass


@register(Decoder)
class DecoderCGP(Decoder):
    def __init__(
        self,
        n_inputs: int,
        n_nodes: int,
        libraries: List[Library],
        endpoint: Endpoint,
    ):
        super().__init__()
        self.adapter = Adapter(
            n_inputs,
            n_nodes,
            returns=endpoint.inputs,
            libraries=libraries,
        )
        self.libraries = libraries
        self.endpoint = endpoint

    def decode_population(
        self, population: Population, x: List[np.ndarray]
    ) -> List:
        y_pred = []
        for i in range(1, population.size):
            y, t = self.decode(population.individuals[i], x)
            population.set_time(i, t)
            y_pred.append(y)
        return y_pred

    def decode(self, genotype: Genotype, x: List[np.ndarray]):
        all_y_pred = []
        all_times = []
        phenotype = self.parse_to_graphs(genotype)
        # for each image
        for xi in x:
            start_time = time.time()
            y_pred = self._decode_one(genotype, phenotype, xi)
            if self.endpoint is not None:
                y_pred = self.endpoint.call(y_pred)
            all_times.append(time.time() - start_time)
            all_y_pred.append(y_pred)
        whole_time = np.mean(np.array(all_times))
        return all_y_pred, whole_time

    def _decode_one(self, genotype: Genotype, phenotype: List, x: List):
        # fill output_map with inputs
        node_outputs = []
        for _ in range(len(self.adapter.types_map)):
            node_outputs.append({})
        outputs = self.adapter.get_outputs(genotype)
        for idx, edge in enumerate(outputs):
            if edge < self.adapter.n_inputs:
                chromosome_idx = self.adapter.types_map[
                    self.adapter.returns[idx]
                ]
                """
                if self.adapter.returns[idx] == TypeFourier:
                    node_outputs[chromosome_idx][edge] = FFT().call(
                        [x[edge]], []
                    )
                elif self.adapter.returns[idx] == Scalar:
                    node_outputs[chromosome_idx][edge] = MeanValue().call(
                        [x[edge]], []
                    )
                
                """
                node_outputs[chromosome_idx][edge] = x[edge]
        self._x_to_output_map(genotype, phenotype, x, node_outputs)
        y = [
            node_outputs[self.adapter.types_map[t]][c]
            for c, t in zip(outputs, self.adapter.returns)
        ]
        return y

    def _x_to_output_map(
        self,
        genotype: Genotype,
        phenotype: List,
        x: List,
        node_outputs,
    ):
        for graph in phenotype:
            for node in graph:
                node_index, type_index = node
                if node_index < self.adapter.n_inputs:
                    continue
                if (
                    node_index
                    in node_outputs[self.adapter.types_map[type_index]].keys()
                ):
                    continue
                real_index = node_index - self.adapter.n_inputs
                # fill the map with active nodes
                function_index = self.adapter.get_function(
                    genotype, type_index, real_index
                )
                p = self.adapter.get_parameters(
                    genotype, type_index, real_index
                )
                arity = self.arity_of(type_index, function_index)
                connections = self.adapter.get_active_edges(
                    genotype, type_index, real_index, arity
                )
                function_input_types = self.inputs_of(
                    type_index, function_index
                )
                inputs = []
                for c, t in zip(connections, function_input_types):
                    chromosome = self.adapter.types_map[t]
                    # TODO: make it modular
                    """
                    if c < self.adapter.n_inputs:
                        if t == TypeFourier:
                            inputs.append(FFT().call([x[c]], []))
                        elif t == Scalar:
                            inputs.append(MeanValue().call([x[c]], []))
                            # inputs.append(0)
                        else:
                            inputs.append(x[c])
                    else:
                    """
                    if c < self.adapter.n_inputs:
                        inputs.append(x[c])
                    else:
                        inputs.append(node_outputs[chromosome][c])

                value = self.execute(type_index, function_index, inputs, p)
                node_outputs[self.adapter.types_map[type_index]][
                    node_index
                ] = value
        return node_outputs

    def parse_to_graphs(self, genotype: Genotype):
        outputs = self.adapter.get_outputs(genotype)
        graphs_list = []
        for output, chromosome in zip(outputs, self.adapter.returns):
            root = {(output, chromosome)}
            graphs_list.append(self._parse_one_graph(genotype, root))
        return graphs_list

    def get_library(self, chromosome):
        return self.libraries[self.adapter.types_map[chromosome]]

    def arity_of(self, chromosome, function):
        return self.get_library(chromosome).arity_of(function)

    def name_of(self, chromosome, function):
        return self.get_library(chromosome).name_of(function)

    def inputs_of(self, chromosome, function):
        return self.get_library(chromosome).inputs_of(function)

    def execute(self, chromosome, function, inputs, parameters):
        return self.get_library(chromosome).execute(
            function, inputs, parameters
        )

    def _parse_one_graph(self, genotype: Genotype, graph_source):
        next_indices = graph_source.copy()
        output_tree = graph_source.copy()
        while next_indices:
            next_element = next_indices.pop()
            next_index, next_type_index = next_element
            if next_index < self.adapter.n_inputs:
                continue
            node = next_index - self.adapter.n_inputs
            function_index = self.adapter.get_function(
                genotype, next_type_index, node
            )
            arity = self.arity_of(next_type_index, function_index)
            types = self.inputs_of(next_type_index, function_index)
            next_connections = self.adapter.get_active_edges(
                genotype, next_type_index, node, arity
            )
            next_connections_to_pop = set(zip(next_connections, types))
            next_indices = next_indices.union(next_connections_to_pop)
            output_tree = output_tree.union(next_connections_to_pop)
        return sorted(list(output_tree))

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "DecoderCGP":
        n_inputs = dict_infos["adapter"]["n_inputs"]
        n_nodes = dict_infos["adapter"]["n_nodes"]
        libraries = [
            Library.__from_dict__(lib_infos)
            for lib_infos in dict_infos["libraries"].values()
        ]
        endpoint = Endpoint.__from_dict__(dict_infos["endpoint"])
        return DecoderCGP(
            n_inputs, n_nodes, libraries=libraries, endpoint=endpoint
        )

    def __to_dict__(self) -> Dict:
        return {
            "adapter": dump_component(self.adapter),
            "libraries": {
                lib.rtype: dump_component(lib) for lib in self.libraries
            },
            "endpoint": dump_component(self.endpoint),
        }
