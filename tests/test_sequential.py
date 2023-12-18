import unittest

from kartezio.core.sequential import ModelSequential
from kartezio.fitness import FitnessIOU
from kartezio.vision.primitives import library_opencv


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        draft = ModelSequential(3, 30, library_opencv, FitnessIOU(reduction="mean"))
        self.model = draft.compile(200, 4, callbacks=[])

    def test_something(self):
        self.assertEqual(self.model.decoder.adapter.n_inputs, 3)
        self.assertEqual(self.model.decoder.adapter.n_outputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_nodes, 30)


if __name__ == "__main__":
    unittest.main()
