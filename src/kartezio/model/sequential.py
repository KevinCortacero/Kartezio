from kartezio.model.base import ModelDraft
from kartezio.model.components.decoder import SequentialDecoder
from kartezio.model.components.endpoint import Endpoint
from kartezio.model.components.library import Library
from kartezio.model.evolution import Fitness


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
