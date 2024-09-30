import unittest

from kartezio.core.components.base import Components
from kartezio.core.sequential import create_sequential_builder
from kartezio.fitness import FitnessIOU
from kartezio.libraries.array import library_opencv
from kartezio.endpoint import EndpointThreshold


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        draft = create_sequential_builder(
            1,
            5,
            library_opencv,
            FitnessIOU(),
            EndpointThreshold(128)
        )
        Components.display()
        self.model = draft.compile(200, 4, callbacks=[])

    def test_something(self):
        self.assertEqual(self.model.decoder.adapter.n_inputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_outputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_nodes, 5)


if __name__ == "__main__":
    unittest.main()
