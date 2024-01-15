import unittest

from kartezio.model.components.base import Component, register, Components


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
        self.assertEqual(Components.contains(Geometry, "circle"), True)
        self.assertEqual(Components.contains(Geometry, "triangle"), False)
        self.assertEqual(Components.contains(Line, "line"), False)
        register(Line, "line")(Line)
        self.assertEqual(Components.contains(Line, "line"), False)


if __name__ == "__main__":
    unittest.main()
