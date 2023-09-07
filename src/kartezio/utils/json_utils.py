import ast
from typing import List

import numpy as np
import simplejson

from kartezio.model.components import GenomeShape, KartezioGenome

""" KartezioGenome Metadata """


def to_metadata(json_data):
    return GenomeShape(
        json_data["n_in"],
        json_data["columns"],
        json_data["n_out"],
        json_data["n_conn"],
        json_data["n_para"],
    )


def to_genome(json_data):
    sequence = np.asarray(ast.literal_eval(json_data["sequence"]))
    return KartezioGenome(sequence=sequence)


def from_individual(individual):
    return {
        "sequence": simplejson.dumps(individual.sequence.tolist()),
        "fitness": individual.fitness,
    }


def from_population(population: List):
    json_data = []
    for individual_idx, individual in population:
        json_data.append(from_individual(individual))
    return json_data


def from_dataset(dataset):
    return {
        "name": dataset.name,
        "label_name": dataset.label_name,
        "indices": dataset.indices,
    }
