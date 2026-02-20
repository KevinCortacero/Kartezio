import unittest

import numpy as np

from kartezio.core.components import Library, Primitive, register
from kartezio.types import DataList, DataType, Parameters, Scalar2


@register(Primitive)
class PrimitiveTest(Primitive):
    def __init__(self):
        super().__init__(Scalar2, DataType.SCALAR, 0)

    def call(self, x: DataList, args: Parameters):
        return x[0] + x[1]


@register(Library)
class LibraryTest(Library):
    pass


class MyTestCase(unittest.TestCase):
    def test_add_primitive(self):
        library = LibraryTest(DataType.SCALAR)
        self.assertEqual(library.size, 0)
        library.display()
        library.add_primitive(PrimitiveTest())
        self.assertEqual(library.size, 1)
        library.display()
        self.assertEqual(library.execute(0, [np.array([21]), np.array([21])], []), 42)

    def test_wrong_return_type(self):
        library = LibraryTest(DataType.MATRIX)
        self.assertEqual(library.size, 0)
        with self.assertRaises(AssertionError):
            library.add_primitive(PrimitiveTest())


if __name__ == "__main__":
    unittest.main()
