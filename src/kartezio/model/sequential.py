from kartezio.improc.primitives import LibraryDefaultOpenCV, library_opencv
from kartezio.model.base import ModelBase
from kartezio.model.components import (
    Library,
    Endpoint,
    DecoderSequential,
)


class ModelSequential(ModelBase):
    def __init__(
        self, inputs: int, nodes: int, library: Library, endpoint: Endpoint = None
    ):
        super().__init__(DecoderSequential(inputs, nodes, library, endpoint))


if __name__ == "__main__":
    model = ModelSequential(2, 30, library_opencv)
