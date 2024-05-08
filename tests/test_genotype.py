import unittest

from kartezio.core.components.base import Components
from kartezio.core.sequential import ModelSequential
from kartezio.fitness import FitnessIOU
from kartezio.vision.primitives import library_opencv


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        draft = ModelSequential(
            1, 5, library_opencv, FitnessIOU(reduction="mean")
        )
        Components.display()
        self.model = draft.compile(200, 4, callbacks=[])

    def test_something(self):
        self.assertEqual(self.model.decoder.adapter.n_inputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_outputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_nodes, 5)


if __name__ == "__main__":
    unittest.main()
