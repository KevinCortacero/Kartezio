import cv2

from kartezio.evolution.decoder import DecoderCGP


class PythonClassWriter:
    def __init__(self, decoder: DecoderCGP):
        self.decoder = decoder
        self.indent_1 = " " * 4
        self.indent_2 = self.indent_1 * 2
        self.imports = "from kartezio.inference import CodeModel\n"
        # endpoint_kwargs = self.decoder.endpoint._to_json_kwargs()
        # import_package = str(self.endpoint.__class__).split("'")[1]
        # import_package = import_package.replace(f".{endpoint_class_name}", "")
        # self.imports += f"from {import_package} import {endpoint_class_name}\n"
        # self.endpoint_instantiation = f"{endpoint_class_name}(**{endpoint_kwargs})"

    def to_python_class(self, class_name, genotype):
        python_code = ""
        python_code += f"{self.imports}\n\n\n"
        python_code += f"class {class_name}(CodeModel):\n"
        # init method
        python_code += f"{self.indent_1}def __init__(self):\n"
        #  python_code += f"{self.indent_2}super().__init__(endpoint={self.endpoint_instantiation})\n\n"
        python_code += "\n"
        # parse method
        python_code += f"{self.indent_1}def _parse(self, X):\n"
        list_of_inputs = []
        map_of_input = {}
        list_of_nodes = []
        map_of_nodes = {}
        list_of_outputs = []
        map_of_outputs = {}
        graphs = self.decoder.parse_to_graphs(genotype)
        print(graphs)
        for i in range(self.decoder.adapter.n_outputs):
            active_nodes = graphs[i][0]
            print(active_nodes)
            for node_infos in active_nodes:
                print(node_infos)
                node, chromosome = node_infos
                if node in list_of_inputs or node in list_of_nodes:
                    continue
                if node < self.decoder.adapter.n_inputs:
                    list_of_inputs.append(node)
                    map_of_input[node] = (
                        f"{self.indent_2}{chromosome}_{node} = X[{node}]\n"
                    )
                elif node < self.decoder.adapter.out_idx:
                    function_index = self.decoder.adapter.get_function(
                        genotype,
                        "chromosome_0",
                        chromosome,
                        node - self.decoder.adapter.n_inputs,
                    )
                    active_edges = self.decoder.arity_of(
                        chromosome, function_index
                    )
                    edges = self.decoder.adapter.get_active_edges(
                        genotype,
                        "chromosome_0",
                        chromosome,
                        node - self.decoder.adapter.n_inputs,
                        active_edges,
                    )
                    parameters = self.decoder.adapter.get_parameters(
                        genotype,
                        "chromosome_0",
                        chromosome,
                        node - self.decoder.adapter.n_inputs,
                    )
                    f_name = self.decoder.name_of(chromosome, function_index)
                    c_types = self.decoder.inputs_of(
                        chromosome, function_index
                    )
                    c_names = [
                        (
                            f"{ctype}_{edge}"
                            if edge < self.decoder.adapter.n_inputs
                            else f"{ctype}_{edge}"
                        )
                        for edge, ctype in zip(edges, c_types)
                    ]
                    c_names = "[" + ", ".join(c_names) + "]"
                    list_of_nodes.append(node)
                    map_of_nodes[node] = (
                        f'{self.indent_2}{chromosome}_{node} = self.call_node("{f_name}", {c_names}, {list(parameters)})\n'
                    )
            list_of_outputs.append(i)
            map_of_outputs[i] = f"{self.indent_2}y_{i} = {chromosome}_{node}\n"
        for input_node in sorted(set(list_of_inputs)):
            python_code += map_of_input[input_node]
        for function_node in sorted(set(list_of_nodes)):
            python_code += map_of_nodes[function_node]
        for output_node in sorted(set(list_of_outputs)):
            python_code += map_of_outputs[output_node]
        output_list = str(
            [f"y_{y}" for y in range(self.decoder.adapter.n_outputs)]
        ).replace("'", "")
        output_list = f"{self.indent_2}Y = {output_list}\n"
        python_code += output_list
        python_code += f"{self.indent_2}return Y\n"
        print()
        print(f"# {'=' * 30} GENERATED CODE TO COPY {'=' * 32}")
        print(python_code)
        print(f"# {'=' * 86}")


class KartezioInsight(DecoderCGP):
    def __init__(self, parser: DecoderCGP, preprocessing=None):
        super().__init__(parser.infos, parser.library, parser.endpoint)
        self.preprocessing = preprocessing

    def create_node_images(self, genotype, x, prefix="", crop=None):
        if self.preprocessing:
            x = self.preprocessing.call([x])[0]
        graphs = self.parse_to_graphs(genotype)
        output_map = self._x_to_output_map(genotype, graphs, x)
        outputs = self._parse_one(genotype, graphs, x)
        endpoint_output = self.endpoint.call(outputs)
        for node_name, node_image in output_map.items():
            if crop:
                node_image = node_image[
                    crop[1] : crop[1] + crop[3], crop[0] : crop[0] + crop[2]
                ]
            heatmap_color = cv2.applyColorMap(node_image, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(f"{prefix}_node_{node_name}.png", heatmap_color)
        output_labels = endpoint_output["labels"]
        if crop:
            output_labels = output_labels[
                crop[1] : crop[1] + crop[3], crop[0] : crop[0] + crop[2]
            ]
        heatmap_color = cv2.applyColorMap(
            output_labels.astype("uint8") * 5, cv2.COLORMAP_VIRIDIS
        )
        cv2.imwrite(f"{prefix}_output.png", heatmap_color)
