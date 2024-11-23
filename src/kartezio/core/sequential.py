from typing import List

from kartezio.core.base import ModelBuilder
from kartezio.core.decoder import DecoderPoly, SequentialDecoder
from kartezio.components.endpoint import Endpoint
from kartezio.components.initializer import MutationAllRandomPoly
from kartezio.components.library import Library
from kartezio.core.evolution import Fitness
from kartezio.mutation.base import MutationRandom


def create_model_builder(
    n_inputs: int,
    n_nodes: int,
    libraries: Library,
    fitness: Fitness,
    endpoint: Endpoint,
    behavior=None,
):
    if not isinstance(libraries, list):
        libraries = [libraries]
    decoder = DecoderPoly(n_inputs, n_nodes, libraries, endpoint)
    return ModelBuilder(
        decoder,
        fitness,
        init=MutationAllRandomPoly(decoder.adapter),
        mutation=MutationRandom(decoder.adapter, 0.15, 0.2),
        behavior=behavior,
    )


def create_poly_builder(
    n_inputs: int,
    n_nodes: int,
    libraries: Library,
    fitness: Fitness,
    endpoint: Endpoint,
    behavior=None,
):
    decoder = DecoderPoly(n_inputs, n_nodes, libraries, endpoint)
    return ModelBuilder(
        decoder,
        fitness,
        init=MutationAllRandomPoly(decoder),
        mutation=MutationRandomPoly(decoder, 0.15, 0.2),
        behavior=behavior,
    )
