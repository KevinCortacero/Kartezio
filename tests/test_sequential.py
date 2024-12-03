import unittest

from kartezio.components.core import Components
from kartezio.endpoint import EndpointThreshold
from kartezio.evolution.base import KartezioSequentialTrainer
from kartezio.fitness import FitnessIOU
from kartezio.libraries.array import create_array_lib
from skimage.data import cell


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        lib = create_array_lib()
        self.model = KartezioSequentialTrainer(
            1, 30, lib, EndpointThreshold(128), FitnessIOU()
        )
        image_x = cell().copy()
        image_y = lib.execute(12, [image_x.copy()], [128])
        image_y = lib.execute(0, [image_x, image_y], [])
        self.x = [[image_x]]
        self.y = [[image_y]]

    def test_something(self):
        self.assertEqual(self.model.decoder.adapter.n_inputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_outputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_nodes, 30)
        self.model.fit(200, self.x, self.y)
        # self.model.print_python_class("Test")


if __name__ == "__main__":
    unittest.main()
