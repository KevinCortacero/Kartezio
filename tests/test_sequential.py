import unittest

from kartezio.fitness import FitnessIOU
from kartezio.improc.primitives import library_opencv
from kartezio.model.sequential import ModelSequential


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        draft = ModelSequential(3, 30, library_opencv, FitnessIOU(reduction="mean"))
        self.model = draft.compile(200, 4, callbacks=[])

    def test_something(self):
        self.assertEqual(self.model.decoder.infos.n_inputs, 3)
        self.assertEqual(self.model.decoder.infos.n_outputs, 1)
        self.assertEqual(self.model.decoder.infos.n_nodes, 30)


if __name__ == "__main__":
    unittest.main()
