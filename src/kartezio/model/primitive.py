from dataclasses import dataclass, field
from typing import Sequence, Callable

from kartezio.model.types import KType


@dataclass
class KSignature:
    name: str
    input_types: Sequence[KType]
    output_type: KType
    n_parameters: int = 0
    arity: int = field(init=False)

    def __post_init__(self):
        self.arity = len(self.input_types)

class KPrimitive:
    def __init__(self, symbol: str, signature: KSignature, function: Callable):
        self.symbol = symbol
        self.signature = signature
        self.function = function
        assert callable(
            self.function
        ), f"given 'function' {self.signature.name} is not callable! (type: {type(function)})"

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
