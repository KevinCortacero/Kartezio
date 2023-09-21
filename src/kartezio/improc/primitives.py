from dataclasses import dataclass, field
from typing import List, Callable, Sequence

from kartezio.model.primitive import KPrimitive, KSignature
from kartezio.model.types import KType, TypeArray

@dataclass
class SignatureImage(KSignature):
    output_type: KType = field(init=False, default=TypeArray)


@dataclass
class SignatureOneImage(SignatureImage):
    input_types: Sequence[KType] = field(init=False, default_factory=lambda: [TypeArray])


@dataclass
class SignatureTwoImage(SignatureImage):
    input_types: Sequence[KType] = field(init=False, default_factory=lambda: [TypeArray, TypeArray])


if __name__ == '__main__':
    signature = SignatureImage("test", [TypeArray], n_parameters=0)
    signature_1 = SignatureOneImage("test_1", n_parameters=2)
    signature_2 = SignatureTwoImage("test_2", n_parameters=0)
    print(signature)
    print(signature_1)
    print(signature_2)