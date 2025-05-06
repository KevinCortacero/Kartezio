from typing import Dict

import numpy as np

from kartezio.core.components import (
    Components,
    Endpoint,
    dump_component,
    load_component,
    register,
)
from kartezio.types import Vector


@register(Endpoint)
class YourEndpoint(Endpoint):
    """
    A custom endpoint used as the final output node in a CGP graph.

    This endpoint calculates the mean of input arrays and returns the index of the class with the highest mean value.
    """

    def __init__(self, n_classes: int):
        """
        Initialize the YourEndpoint with the specified number of classes.

        Args:
            n_classes (int): The number of classes to handle as inputs.
        """
        super().__init__(inputs=[Vector] * n_classes)
        self.n_classes = n_classes

    def call(self, x, args=None) -> int:
        """
        Compute the index of the class with the highest mean value from the input arrays.

        Args:
            x (List[np.ndarray]): A list of input arrays, one for each class.
            args (optional): Additional arguments, not used in this endpoint.

        Returns:
            int: The index of the class with the highest mean value.
        """
        return np.argmax(np.mean(np.array(x), axis=1))

    def __to_dict__(self) -> Dict:
        """
        Convert the endpoint instance to a dictionary representation.

        Returns:
            Dict: A dictionary containing the arguments needed to recreate the endpoint.
        """
        return {
            "args": {"n_classes": self.n_classes},
        }


def main():
    """
    Demonstrate the use of the custom endpoint by processing a sample set of input arrays.
    """
    # Create an instance of the custom endpoint
    your_endpoint = YourEndpoint(n_classes=3)

    # Define sample input arrays
    inputs = [
        np.array([0, 0, 42, 44]),
        np.array([0, 43, 42, 0]),
        np.array([1, 2, 3, 44]),
    ]

    # Apply the endpoint operation
    output = your_endpoint.call(inputs)  # Expected Output: 0
    print(f"Output from your_endpoint: {output}")

    # Instantiate the endpoint from the component registry and apply it
    your_endpoint_2 = Components.instantiate(
        "Endpoint", "YourEndpoint", n_classes=3
    )
    output_2 = your_endpoint_2.call(inputs)  # Expected Output: 0
    print(f"Output from your_endpoint_2: {output_2}")

    # Dump the component as a dictionary
    print("Component Dump:", dump_component(your_endpoint))

    your_endpoint_3 = load_component(Endpoint, dump_component(your_endpoint))
    output_3 = your_endpoint_3.call(inputs)  # Expected Output: 0
    print(f"Output from your_endpoint_3: {output_3}")


if __name__ == "__main__":
    main()
