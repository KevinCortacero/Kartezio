import unittest

from kartezio.components.core import Component, Components, register


class Geometry(Component):
    pass


class Line(Geometry):
    pass


@register(Geometry, "circle")
class Circle(Geometry):
    pass


@register(Geometry, "square")
class Square(Geometry):
    pass


@register(Geometry, "square", replace=True)
class AnotherSquare(Geometry):
    pass


class TestComponents(unittest.TestCase):
    def test_contains(self):
        self.assertTrue(Components.contains(Geometry, "circle"))
        self.assertFalse(Components.contains(Geometry, "triangle"))
        self.assertFalse(Components.contains(Line, "line"))
        register(Line, "line")(Line)
        self.assertTrue(Components.contains(Line, "line"))


if __name__ == "__main__":
    unittest.main()
