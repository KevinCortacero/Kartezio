import unittest

from skimage.data import cell

from kartezio.core.endpoints import EndpointThreshold
from kartezio.core.fitness import IoU
from kartezio.evolution.base import KartezioTrainer
from kartezio.primitives.matrix import create_array_lib


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        lib = create_array_lib()
        self.model = KartezioTrainer(1, 30, lib, EndpointThreshold(128), IoU())
        self.model.set_mutation_rates(0.1, 0.1)
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
        self.model.print_python_class("Test")


if __name__ == "__main__":
    unittest.main()
