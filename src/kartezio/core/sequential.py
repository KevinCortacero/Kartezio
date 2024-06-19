from kartezio.core.base import ModelBuilder
from kartezio.core.components.decoder import SequentialDecoder
from kartezio.core.components.endpoint import Endpoint
from kartezio.core.components.library import Library
from kartezio.core.evolution import Fitness


def create_sequential_builder(n_inputs: int, n_nodes: int, library: Library, fitness: Fitness, endpoint: Endpoint, behavior=None):
    return ModelBuilder(SequentialDecoder(n_inputs, n_nodes, library, endpoint), fitness, behavior=behavior)
    
