import unittest

from kartezio.core.endpoints import EndpointThreshold
from kartezio.core.fitness import IoU
from kartezio.evolution.base import KartezioTrainer
from kartezio.primitives.matrix import default_matrix_lib


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = KartezioTrainer(
            1,
            5,
            default_matrix_lib(),
            EndpointThreshold(128),
            IoU(),
        )

    def test_something(self):
        self.assertEqual(self.model.decoder.adapter.n_inputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_outputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_nodes, 5)


if __name__ == "__main__":
    unittest.main()
