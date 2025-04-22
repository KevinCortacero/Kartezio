import unittest

from kartezio.core.components import (
    Components,
    KartezioComponent,
    fundamental,
    register,
)


@fundamental()
class Geometry(KartezioComponent):
    pass


class Line(Geometry):
    pass


@register(Geometry)
class Circle(Geometry):
    pass


@register(Geometry)
class Square(Geometry):
    pass


@register(Geometry, replace=Square)
class AnotherSquare(Geometry):
    pass


class TestComponents(unittest.TestCase):
    def test_contains(self):
        self.assertTrue(Components.contains(Geometry, "Circle"))
        self.assertFalse(Components.contains(Geometry, "Triangle"))
        self.assertFalse(Components.contains(Geometry, "Line"))
        self.assertFalse(Components.contains(Geometry, "AnotherSquare"))
        self.assertTrue(Components.contains(Geometry, "Square"))
        self.assertTrue(
            AnotherSquare == Components._registry["Geometry"]["Square"]
        )
        register(Geometry)(Line)
        self.assertTrue(Components.contains(Geometry, "Line"))


if __name__ == "__main__":
    unittest.main()
