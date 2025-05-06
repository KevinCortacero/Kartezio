from io import BytesIO

import networkx as nx
import pygraphviz as pgv
from networkx.drawing.nx_agraph import from_agraph
from PIL import Image

from kartezio.evolution.decoder import Decoder


class KartezioViewer(Decoder):
    def __init__(self, metadata, bundle, endpoint):
        super().__init__(metadata, bundle, endpoint)

    def get_graph(
        self,
        genome,
        inputs=None,
        outputs=None,
        only_active=True,
        jupyter=False,
    ):
        G = pgv.AGraph(name="genome", directed="true")
        G.graph_attr["rankdir"] = "LR"
        G.graph_attr["ranksep"] = 0.75
        G.graph_attr["splines"] = "true"
        G.graph_attr["compound"] = "true"

        G.node_attr["fontsize"] = "14"
        G.node_attr["style"] = "filled"
        G.node_attr["shape"] = "ellipse"
        G.node_attr["fillcolor"] = "white"
        G.node_attr["fixedsize"] = "true"
        G.node_attr["width"] = 1.61803398875
        G.node_attr["height"] = 1.0

        if only_active:
            active_nodes = self.parse_to_graphs(genome)
            active_nodes = sum(active_nodes, [])

        with G.subgraph(
            range(self.infos.inputs),
            name="cluster_inputs",
            penwidth=0,
            rank="source",
        ) as cluster_inputs:
            for node in range(self.infos.inputs):
                if inputs:
                    label = f"- {node} -\n{inputs[node]}"
                else:
                    label = f"- {node} -\nIN_{node}"
                cluster_inputs.add_primitive(
                    node, label=label, fillcolor="#B6D7A8"
                )

        with G.subgraph(
            range(self.infos.inputs, self.infos.out_idx),
            name="cluster_genes",
            penwidth=0,
        ) as cluster_genes:
            for node in range(self.infos.inputs, self.infos.out_idx):
                if only_active and node not in active_nodes:
                    continue
                function_index = self.read_function(
                    genome, node - self.infos.inputs
                )
                function_name = self.library.f_name_of(function_index)
                cluster_genes.add_primitive(
                    node, label=f"- {node} -\n{function_name}"
                )
                active_connections = self.library.arity_of(function_index)
                connections = self.read_active_connections(
                    genome, node - self.infos.inputs, active_connections
                )
                # parameters = self.read_parameters(genome, node - self.infos.inputs)

                for c in connections:
                    G.add_edge(c, node)

        with G.subgraph(
            range(self.infos.out_idx, self.infos.out_idx + self.infos.outputs),
            name="cluster_outputs",
            penwidth=0,
            rank="sink",
        ) as cluster_outputs:
            for node in range(
                self.infos.out_idx, self.infos.out_idx + self.infos.outputs
            ):
                c = self.read_outputs(genome)[node - self.infos.out_idx][
                    self.infos.con_idx
                ]
                if outputs:
                    label = f"- {node} -\n{outputs[node - self.infos.out_idx]}"
                else:
                    label = f"- {node} -\nOUT_{node}"
                cluster_outputs.add_primitive(
                    node, label=label, fillcolor="#F9CB9C"
                )
                G.add_edge(c, node)

        # converting pygraphviz graph to networkx graph
        X = from_agraph(G)
        max_rank = 0
        node_rank = 0
        for node_in in range(self.infos.inputs):
            # dictionary {node: length}
            lengths = nx.shortest_path_length(X, str(node_in))
            result = max(lengths.items(), key=lambda p: p[1])
            if result[1] > max_rank:
                max_rank = result[1]
                node_rank = result[0]

        for node_out in range(
            self.infos.out_idx, self.infos.out_idx + self.infos.outputs
        ):
            G.add_edge(node_rank, node_out, style="invis")

        G.layout(prog="dot")
        if jupyter:
            # create image without saving to disk
            return Image.open(BytesIO(G.draw(format="png")))
        return G
