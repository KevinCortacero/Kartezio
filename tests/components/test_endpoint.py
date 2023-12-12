import unittest
from typing import Dict, List

from kartezio.model.components.base import Components, register
from kartezio.model.components.endpoint import Endpoint
from kartezio.model.types import TypeScalar


@register(Endpoint, "add_value")
class EndpointTest(Endpoint):
    def call(self, x) -> List:
        return [x[0] + self.value]

    def __init__(self, value):
        super().__init__([TypeScalar])
        self.value = value

    def __to_dict__(self) -> Dict:
        return {"args": [self.value]}


class MyTestCase(unittest.TestCase):
    def test_something(self):
        endpoint = EndpointTest(42)
        self.assertEqual(endpoint.call([42])[0], 84)
        endpoint_2 = Components.instantiate("Endpoint", "add_value", 42)
        self.assertEqual(endpoint_2.call([42])[0], 84)
        dict_infos = {"name": endpoint.name, **endpoint.__to_dict__()}
        endpoint_3 = Endpoint.__from_dict__(dict_infos=dict_infos)
        self.assertEqual(endpoint_3.call([42])[0], 84)


if __name__ == "__main__":
    unittest.main()
