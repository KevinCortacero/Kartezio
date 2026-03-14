import unittest

from kartezio.core.components import Components, Endpoint, register
from kartezio.types import DataType


@register(Endpoint)
class EndpointTest(Endpoint):
    def __init__(self, value: int):
        super().__init__([DataType.SCALAR])
        self.value = value

    def call(self, x) -> list:
        return [x[0] + self.value]

    def __to_dict__(self) -> dict:
        return {"name": self.name, "args": [self.value]}


class MyTestCase(unittest.TestCase):
    def test_something(self):
        endpoint = EndpointTest(42)
        self.assertEqual(endpoint.call([42])[0], 84)
        endpoint_2 = Components.instantiate("Endpoint", "EndpointTest", 21)
        self.assertEqual(endpoint_2.call([21])[0], 42)


if __name__ == "__main__":
    unittest.main()
