from kartezio.core.base import ModelDraft
from kartezio.core.components.decoder import SequentialDecoder
from kartezio.core.components.endpoint import Endpoint
from kartezio.core.components.library import Library
from kartezio.core.evolution import Fitness


class ModelSequential(ModelDraft):
    def __init__(
        self,
        n_inputs: int,
        n_nodes: int,
        library: Library,
        fitness: Fitness,
        endpoint: Endpoint = None,
    ):
        super().__init__(
            SequentialDecoder(n_inputs, n_nodes, library, endpoint), fitness
        )
