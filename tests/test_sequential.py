import unittest

from skimage.data import cell

from kartezio.core.components.base import Components
from kartezio.core.sequential import create_model_builder
from kartezio.endpoint import EndpointThreshold
from kartezio.fitness import FitnessIOU
from kartezio.libraries.array import library_opencv


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        builder = create_model_builder(
            1,
            30,
            library_opencv,
            FitnessIOU(),
            EndpointThreshold(128)
        )
        self.model = builder.compile(200, 4, callbacks=[])
        image_x = cell().copy()
        image_y = library_opencv.execute(12, [image_x.copy()], [128])
        image_y = library_opencv.execute(0, [image_x, image_y], [])
        self.x = [[image_x]]
        self.y = [[image_y]]

    def test_something(self):
        self.assertEqual(self.model.decoder.adapter.n_inputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_outputs, 1)
        self.assertEqual(self.model.decoder.adapter.n_nodes, 30)
        self.model.fit(self.x, self.y)
        # self.model.print_python_class("Test")


if __name__ == "__main__":
    unittest.main()
