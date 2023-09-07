import os

from numena.io.json import json_read, json_write
from numena.io.drive import Directory

import kartezio.utils.json_utils as json
from kartezio.model.components import KartezioGenome, KartezioParser


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
        current_generation = json_read(filepath=f"{directory_path}/G{generation}.json")
        generation_json = {
            "generation": generation,
            "population": current_generation["population"]
        }
        packed_history["generations"].append(generation_json)
    json_write(filepath=f"{directory_path}/history.json", json_data=packed_history, indent=None)
    print(f"All generations packed in {directory_path}.")
    for generation in generations:
        file_to_delete = f"{directory_path}/G{generation}.json"
        os.remove(file_to_delete)
    print(f"All {len(generations)} generation files deleted.")


class JsonLoader:
    def read_individual(self, filepath):
        json_data = json_read(filepath=filepath)
        dataset = json_data["dataset"]
        parser = KartezioParser.from_json(json_data["decoding"])
        try:
            individual = KartezioGenome.from_json(json_data["individual"])
        except KeyError:
            try:
                individual = KartezioGenome.from_json(json_data)
            except KeyError:
                individual = KartezioGenome.from_json(json_data["population"][0])
        return dataset, individual, parser


class JsonSaver:
    def __init__(self, dataset, parser):
        self.dataset_json = json.from_dataset(dataset)
        self.parser_as_json = parser.dumps()

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
