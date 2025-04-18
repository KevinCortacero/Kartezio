import os

from kartezio.core.components import (
    Fitness,
    Genotype,
    Preprocessing,
    load_component,
)
from kartezio.evolution.decoder import DecoderCGP
from kartezio.utils.directory import Directory
from kartezio.utils.json_handler import json_read, json_write


def pack_one_directory(directory_path):
    directory = Directory(directory_path)
    packed_history = {}
    elite = json_read(filepath=f"{directory_path}/elite.json")
    packed_history["dataset"] = elite["dataset"]
    packed_history["decoding"] = elite["decoding"]
    packed_history["elite"] = elite["individual"]
    packed_history["generations"] = []
    generations = []
    for g in directory.ls(f"G*.json", ordered=True):
        generations.append(int(g.name.replace("G", "").split(".")[0]))
    generations.sort()
    for generation in generations:
        current_generation = json_read(
            filepath=f"{directory_path}/G{generation}.json"
        )
        generation_json = {
            "generation": generation,
            "population": current_generation["population"],
        }
        packed_history["generations"].append(generation_json)
    json_write(
        filepath=f"{directory_path}/history.json",
        json_data=packed_history,
        indent=None,
    )
    print(f"All generations packed in {directory_path}.")
    for generation in generations:
        file_to_delete = f"{directory_path}/G{generation}.json"
        os.remove(file_to_delete)
    print(f"All {len(generations)} generation files deleted.")


class JsonLoader:
    def read_individual(self, filepath):
        json_data = json_read(filepath=filepath)
        dataset = json_data["dataset"]
        decoder = load_component(DecoderCGP, json_data["decoder"])
        if decoder is None:
            raise ValueError("Decoder not found.")
        individual = load_component(Genotype, json_data["elite"])
        if json_data["preprocessing"] is None:
            preprocessing = None
        else:
            preprocessing = load_component(
                Preprocessing, json_data["preprocessing"]
            )
        fitness = load_component(Fitness, json_data["fitness"])
        return dataset, individual, decoder, preprocessing, fitness

class JsonLoader3D:
    def read_individual(self, filepath):
        json_data = json_read(filepath=filepath)
        dataset = json_data["dataset"]
        decoder_type = json_data["decoder"]["name"]
        decoder_class = globals().get(decoder_type)
        if decoder_class is None:
            raise ValueError(f"Decoder class '{decoder_class}' not found.")
        decoder = load_component(decoder_class, json_data["decoder"])
        if decoder is None:
            raise ValueError("Decoder not found.")
        individual = load_component(Genotype, json_data["elite"])
        if json_data["preprocessing"] is None:
            preprocessing = None
        else:
            preprocessing = load_component(
                Preprocessing, json_data["preprocessing"]
            )
        fitness = load_component(Fitness, json_data["fitness"])
        return dataset, individual, decoder, preprocessing, fitness

"""
class JsonSaver:
    def __init__(self, dataset, parser: Decoder):
        self.dataset_json = json.from_dataset(dataset)
        self.parser_as_json = parser.to_toml()

    def save_population(self, filepath, population):
        json_data = {
            "dataset": self.dataset_json,
            "population": json.from_population(population),
            "decoding": self.parser_as_json,
        }
        json_write(filepath, json_data)

    def save_individual(self, filepath, individual):
        json_data = {
            "dataset": self.dataset_json,
            "individual": json.from_individual(individual),
            "decoding": self.parser_as_json,
        }
        json_write(filepath, json_data)
"""
