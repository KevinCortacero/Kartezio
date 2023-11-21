from kartezio.model.base import ModelBase
from kartezio.model.components import (
    Library,
    Endpoint,
    DecoderSequential,
)
from kartezio.model.evolution import Fitness


class ModelSequential(ModelBase):
    def __init__(
        self,
        n_inputs: int,
        n_nodes: int,
        library: Library,
        fitness: Fitness,
        endpoint: Endpoint = None,
    ):
        super().__init__(
            DecoderSequential(n_inputs, n_nodes, library, endpoint), fitness
        )
