from dataclasses import dataclass, field
from typing import List, Callable, Sequence

import cv2

from kartezio.model.registry import registry
from kartezio.model.library import EmptyLibrary
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


def f_max(x, args=None):
    return cv2.max(x[0], x[1])


def f_min(x, args=None):
    return cv2.min(x[0], x[1])


p_max = KPrimitive("max", SignatureTwoImage("Max"), f_max)
p_min = KPrimitive("min", SignatureTwoImage("Min"), f_min)
# registry.primitives.add("max", replace=True)(p_max)


if __name__ == '__main__':
    signature = SignatureImage("test", [TypeArray], n_parameters=0)
    signature_1 = SignatureOneImage("test_1", n_parameters=2)
    signature_2 = SignatureTwoImage("test_2", n_parameters=0)
    print(signature)
    print(signature_1)
    print(signature_2)

    library_test = EmptyLibrary(TypeArray)
    library_test.add_primitive(p_max)
    library_test.add_primitive(p_min)
    library_test.display()