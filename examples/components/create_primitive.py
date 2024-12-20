from typing import List

import numpy as np

from kartezio.core.components import Components, Primitive, register
from kartezio.types import TypeArray


@register(Primitive, "my_primitive")
class MyExamplePrimitive(Primitive):
    """
    A custom primitive operation that replaces specific pixel values.

    This primitive replaces pixels equal to the defined value (`self.value`) with 1, and all other pixels with 0.
    """

    def __init__(self):
        """
        Initialize the MyExamplePrimitive with a default value of 42.
        """
        super().__init__([TypeArray], TypeArray, 0)
        self.value = 42

    def call(self, x: List[np.ndarray], args: List[int] = None) -> np.ndarray:
        """
        Replace pixels equal to `self.value` with 1, and the rest with 0.

        Args:
            x (List[np.ndarray]): The input array(s) to be processed.
            args (List[int], optional): Additional arguments, not used in this primitive.

        Returns:
            np.ndarray: The processed array with pixels replaced.
        """
        return (x[0] == self.value).astype(int)


def main():
    """
    Demonstrate the use of the custom primitive by processing a sample input array.
    """
    my_primitive = MyExamplePrimitive()
    inputs = [np.array([42, 43, 42, 24])]  # Sample input array
    output = my_primitive.call(inputs)  # Apply the primitive operation
    print(output)  # Output: [1, 0, 1, 0]

    my_primitive_2 = Components.instantiate("Primitive", "my_primitive")
    output_2 = my_primitive_2.call(inputs)
    print(output_2)  # Output: [1, 0, 1, 0]


if __name__ == "__main__":
    main()
