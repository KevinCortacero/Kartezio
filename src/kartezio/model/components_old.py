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
