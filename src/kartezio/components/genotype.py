import ast
import copy
from typing import Dict

import numpy as np
from kartezio.components.core import Component


class Genotype(Component):
    """
    Represents the genotype for Cartesian Genetic Programming (CGP).

    This class stores the "DNA" in the form of Numpy arrays (`ndarray`). No metadata is included in the DNA to
    prevent duplication, following the Flyweight design pattern.

    Reference:
    Flyweight Pattern - https://refactoring.guru/design-patterns/flyweight
    """

    def __init__(self, n_outputs: int):
        """
        Initialize a Genotype instance with a specified number of outputs.

        Args:
            n_outputs (int): The number of outputs for the genotype.
        """
        super().__init__()
        self._chromosomes: Dict[str, np.ndarray] = {}
        self._chromosomes["outputs"] = np.zeros(n_outputs, dtype=np.uint8)

    def __getitem__(self, item: str) -> np.ndarray:
        """
        Get the chromosome array by key.

        Args:
            item (str): The key representing the chromosome.

        Returns:
            np.ndarray: The chromosome array corresponding to the provided key.
        """
        return self._chromosomes[item]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """
        Set the chromosome array for a given key.

        Args:
            key (str): The key for the chromosome to be set.
            value (np.ndarray): The new value for the chromosome.
        """
        self._chromosomes[key] = value

    def __deepcopy__(self, memo) -> "Genotype":
        """
        Create a deep copy of the genotype.

        Args:
            memo (dict): A dictionary of objects that have already been copied to prevent infinite recursion.

        Returns:
            Genotype: A deep copy of the genotype instance.
        """
        new = self.__class__(len(self._chromosomes["outputs"]))
        for key, value in self._chromosomes.items():
            new._chromosomes[key] = value.copy()
        return new

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Genotype":
        """
        Create a Genotype instance from a dictionary representation.

        Args:
            dict_infos (Dict): A dictionary containing chromosome information.

        Returns:
            Genotype: A new Genotype instance created from the given dictionary.
        """
        assert "chromosomes" in dict_infos, "Expected 'chromosomes' key in dictionary."
        assert (
            "outputs" in dict_infos["chromosomes"]
        ), "Expected 'outputs' key in 'chromosomes' dictionary."
        n_outputs = len(dict_infos["chromosomes"]["outputs"])
        genotype = cls(n_outputs)
        for key, value in dict_infos["chromosomes"].items():
            genotype[key] = np.asarray(value)
        return genotype
    
    def __to_dict__(self) -> Dict:
        """
        Convert the genotype to a dictionary representation.

        Returns:
            Dict: A dictionary containing the chromosome information.
        """
        return {"chromosomes": {key: value.tolist() for key, value in self._chromosomes.items()}}

    def clone(self) -> "Genotype":
        """
        Create a clone of the genotype using deep copy.

        Returns:
            Genotype: A cloned instance of the genotype.
        """
        return copy.deepcopy(self)
