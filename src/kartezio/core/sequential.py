from kartezio.core.base import ModelBuilder
from kartezio.core.components.decoder import DecoderPoly, SequentialDecoder
from kartezio.core.components.endpoint import Endpoint
from kartezio.core.components.initialization import MutationAllRandomPoly
from kartezio.core.components.library import Library
from kartezio.core.evolution import Fitness
from kartezio.mutation import MutationRandomPoly


def create_sequential_builder(
    n_inputs: int,
    n_nodes: int,
    library: Library,
    fitness: Fitness,
    endpoint: Endpoint,
    behavior=None,
):
    return ModelBuilder(
        SequentialDecoder(n_inputs, n_nodes, library, endpoint),
        fitness,
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
