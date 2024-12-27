import unittest

from kartezio.core.components import Components
from kartezio.core.endpoints import EndpointThreshold
from kartezio.core.fitness import FitnessIOU
from kartezio.evolution.base import KartezioTrainer
from kartezio.primitives.array import create_array_lib


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        model = KartezioTrainer(
            1, 5, create_array_lib(), FitnessIOU(), EndpointThreshold(128)
        )
        Components.display()
        # self.model = model.(200, 4, callbacks=[])

    def test_something(self):
        self.assertEqual(self.model.decoder.adapter.n_inputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_outputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_nodes, 5)


if __name__ == "__main__":
    unittest.main()
