import copy
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Dict, List

import numpy as np
from tabulate import tabulate

from kartezio.helpers import Observer
from kartezio.types import KType


class KartezioComponent(ABC):
    """
    Abstract base class for representing a Component in the Kartezio framework.
    This class provides basic functionalities for components.
    """

    def __init__(self):
        """
        Initialize a Component instance.
        The component name is derived using the Components registry.
        """
        self.name = Components.name_of(self.__class__)

    def __to_dict__(self) -> Dict:
        """
        Convert the component to a dictionary representation.

        Returns:
            Dict: An empty dictionary, intended to be overridden by subclasses.
        """
        return {}

    @classmethod
    @abstractmethod
    def __from_dict__(cls, dict_infos: Dict) -> "KartezioComponent":
        """
        Abstract method to instantiate a component from a dictionary representation.

        Args:
            dict_infos (Dict): A dictionary containing information to initialize the component.

        Returns:
            Component: A new component instance.
        """
        pass


class UpdatableComponent(KartezioComponent, Observer, ABC):
    """
    A component that can be updated, typically used in adaptive scenarios.

    Inherits from Component and Observer to enable both observation and update capabilities, making it suitable
    for scenarios where a component's internal state must be adjusted during the evolutionary process.
    """

    def __init__(self):
        """
        Initialize an UpdatableComponent instance.
        """
        super().__init__()
        self.n_iterations = None
        self.stored = None

    def compile(self, n_iterations: int):
        self.n_iterations = n_iterations
        self.stored = self._precompute()

    @abstractmethod
    def _precompute(self):
        pass


class Components:
    _registry = {}
    _reverse = {}

    @staticmethod
    def _contains(parent_name: str, name: str):
        if parent_name not in Components._registry.keys():
            return False
        if name not in Components._registry[parent_name].keys():
            return False
        return True

    @staticmethod
    def contains(parent: type, name: str):
        return Components._contains(parent.__name__, name)

    @staticmethod
    def add_component(component: str):
        if component not in Components._registry.keys():
            Components._registry[component] = {}
        else:
            print(
                f"Fundamental Component '{component}'already found in the registry."
            )

    @staticmethod
    def add(fundamental: str, name: str, component: type):
        assert isinstance(component, type), f"{component} is not a Class!"
        assert issubclass(
            component, KartezioComponent
        ), f"{component} is not a Component! Please inherit from 'Component' or make sure to call 'super().__init__()'."

        if fundamental not in Components._registry.keys():
            raise KeyError(
                f"Fundamental Component '{fundamental}' not found in the registry. Please consider using '@component()' decorator on your '{fundamental}' class"
            )

        if name not in Components._registry[fundamental].keys():
            Components._registry[fundamental][name] = component

        Components._registry[fundamental][name] = component
        Components._reverse[component.__name__] = f"{fundamental}/{name}"

    @staticmethod
    def instantiate(group_name: str, component_name: str, *args, **kwargs):
        if not Components._contains(group_name, component_name):
            raise KeyError(
                f"Component '{group_name}', called '{component_name}' not found in the registry!"
            )
        component = Components._registry[group_name][component_name](
            *args, **kwargs
        )
        return component

    @staticmethod
    def name_of(component_class: type) -> str:
        if component_class.__name__ not in Components._reverse.keys():
            return component_class.__name__
        return Components._reverse[component_class.__name__].split("/")[1]

    @staticmethod
    def list(group_name: str):
        if group_name not in Components._registry.keys():
            raise KeyError(
                f"Fundamental Component '{group_name}' not found in the registry."
            )
        return Components._registry[group_name].keys()

    @staticmethod
    def display():
        pprint(Components._registry)


def register(fundamental: type, replace: type = None):
    """
    Register a component to the Components registry.

    Args:
        fundamental (type): The fundamental type of the component.
        replace (type): If not None, replace an existing component with the type.

    Returns:
        Callable: A decorator for registering the component.
    """
    fundamental_name = fundamental.__name__

    def inner(item_cls):
        name = item_cls.__name__
        if Components._contains(fundamental_name, name):
            if not replace:
                print(
                    f"""Warning registering {fundamental_name} called '{name}'.
                    Here is the list of all registered {fundamental_name} components:
                    \n{Components._registry[fundamental_name].keys()}.
                    \n > Replace it using 'replace=True' in @register, or use another name.
                """
                )
        if replace:
            replace_name = replace.__name__
            if Components._contains(fundamental_name, replace_name):
                print(
                    f"Component '{fundamental_name}/{replace_name}' will be replaced by '{name}'"
                )
                Components.add(fundamental_name, replace_name, item_cls)
        else:
            Components.add(fundamental_name, name, item_cls)
        return item_cls

    return inner


def fundamental():
    """
    Register a fundamental component to the Components registry.

    Returns:
        Callable: A decorator for registering the fundamental component.
    """

    def inner(item_cls):
        Components.add_component(item_cls.__name__)
        return item_cls

    return inner


def load_component(
    component_class: type, json_data: Dict
) -> KartezioComponent:
    """
    Load a component from its dictionary representation.

    Args:
        component_class (type): The class of the component to load.
        json_data (Dict): The dictionary containing component data.

    Returns:
        Component: An instance of the component created from the given data.
    """
    return component_class.__from_dict__(json_data)


def dump_component(component: KartezioComponent) -> Dict:
    """
    Dump a component to its dictionary representation.

    Args:
        component (Component): The component to save.

    Returns:
        Dict: A dictionary representation of the component.
    """
    base_dict = component.__to_dict__()
    base_dict["name"] = component.name
    return base_dict


class Node(KartezioComponent, ABC):
    """
    Abstract base class for a Node in the CGP framework.
    """

    pass


@fundamental()
class Preprocessing(Node, ABC):
    """
    Preprocessing node, called before training loop.
    """

    def __init__(self):
        super().__init__()
        self.__then = None

    def call(self, x, args=None):
        if self.__then is not None:
            return self.__then.call(self.preprocess(x), args)
        return self.preprocess(x)

    @abstractmethod
    def preprocess(self, x):
        raise NotImplementedError

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Preprocessing":
        return Components.instantiate(
            "Preprocessing", dict_infos["name"], **dict_infos["args"]
        )

    def then(self, preprocessing: "Preprocessing"):
        if self.__then is not None:
            self.__then.then(preprocessing)
        else:
            self.__then = preprocessing
        return self


def assert_f32(image):
    if image.dtype != np.float32:
        raise ValueError(
            "Image must be of type float32, not {}".format(image.dtype)
        )


def clip_f32(image):
    return np.clip(image, -1.0, 1.0)


def assert_clipped(image):
    if image.max() > 1.0 or image.min() < -1.0:
        raise ValueError("Image values must be in the range [-1, 1]")


def to_u8(image):
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0, 1).astype(np.float32)
    return (image * 255).astype(np.uint8)


def to_f32(image):
    new_image = image.astype(np.float32) / 255.0
    return clip_f32(new_image)


@fundamental()
class Primitive(Node, ABC):
    """
    Primitive function called inside the CGP Graph.
    """

    def __init__(self, inputs: List[KType], output: KType, n_parameters: int):
        super().__init__()
        self.inputs = inputs
        self.output = output
        self.arity = len(inputs)
        self.n_parameters = n_parameters

    def call_f32(self, x: List[np.ndarray], args: List[int]):
        """
        Call the primitive function with float32 inputs.

        Args:
            x (List[np.ndarray]): List of input arrays.
            args (List[int]): List of arguments for the function.

        Returns:
            np.ndarray: The result of the function call.
        """
        filtered = self.call(x, args)
        clip_f32(filtered)
        return filtered

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Primitive":
        return Components.instantiate("Primitive", dict_infos["name"])

    def __to_dict__(self) -> Dict:
        return {"name": self.name}


@fundamental()
class Genotype(KartezioComponent):
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
        assert (
            "chromosomes" in dict_infos
        ), "Expected 'chromosomes' key in dictionary."
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
        return {
            "chromosomes": {
                key: value.tolist() for key, value in self._chromosomes.items()
            }
        }

    def clone(self) -> "Genotype":
        """
        Create a clone of the genotype using deep copy.

        Returns:
            Genotype: A cloned instance of the genotype.
        """
        return copy.deepcopy(self)


@fundamental()
class Reducer(Node, ABC):
    def batch(self, x: List):
        y = []
        for xi in x:
            y.append(self.reduce(xi))
        return y

    @abstractmethod
    def reduce(self, x):
        pass


@fundamental()
class Endpoint(Node, ABC):
    """
    Represents the final node in a CGP graph, responsible for producing the final outputs.

    The Endpoint is invoked in the training loop but is not involved in the evolutionary process.
    """

    def __init__(self, inputs: List[KType]):
        """
        Initialize an Endpoint instance.

        Args:
            inputs (List[KType]): The list of inputs for the endpoint node.
        """
        super().__init__()
        self.inputs = inputs

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Endpoint":
        """
        Create an Endpoint instance from a dictionary representation.

        Args:
            dict_infos (Dict): A dictionary containing the name and arguments for the Endpoint.

        Returns:
            Endpoint: A new Endpoint instance created from the given dictionary.
        """
        return Components.instantiate(
            "Endpoint",
            dict_infos["name"],
            **dict_infos["args"],
        )

    @classmethod
    def from_config(cls, config):
        return Components.instantiate(
            cls.__name__, config["name"], **config["args"]
        )


@fundamental()
class Fitness(KartezioComponent, ABC):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.mode = "train"

    def batch(self, y_true, y_pred, reduction=None):
        population_fitness = np.zeros(
            (len(y_pred), len(y_true)), dtype=np.float32
        )
        for idx_individual in range(len(y_pred)):
            population_fitness[idx_individual] = self.evaluate(
                y_true, y_pred[idx_individual]
            )
        if self.reduction is not None and reduction is None:
            return self._reduce(population_fitness, self.reduction)
        if reduction is not None:
            return self._reduce(population_fitness, reduction)
        return population_fitness

    def _reduce(self, population_fitness, reduction=None):
        if reduction == "mean":
            return np.mean(population_fitness, axis=1)
        if reduction == "min":
            return np.min(population_fitness, axis=1)
        if reduction == "max":
            return np.max(population_fitness, axis=1)
        if reduction == "median":
            return np.median(population_fitness, axis=1)
        if reduction == "raw":
            return population_fitness

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Fitness":
        from kartezio.core.fitness import Fitness

        return Components.instantiate(
            "Fitness",
            dict_infos["name"],
            **dict_infos["args"],
        )


@fundamental()
class Library(KartezioComponent):
    def __init__(self, rtype):
        super().__init__()
        self._primitives: List[Primitive] = []
        self.rtype = rtype

    def __to_dict__(self) -> Dict:
        return {
            "rtype": self.rtype,
            "primitives": {i: self.name_of(i) for i in range(self.size)},
        }

    @classmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Library":
        rtype = dict_infos["rtype"]
        library = Library(rtype)
        size = len(dict_infos["primitives"])
        for i in range(size):
            library.add_by_name(dict_infos["primitives"][i])
        return library

    def add_by_name(self, name):
        primitive = Components.instantiate("Primitive", name)
        self.add_primitive(primitive)

    def add_class(self, primitive: type):
        self.add_primitive(primitive())

    def add_primitive(self, primitive: Primitive):
        assert isinstance(primitive, Primitive)
        self._primitives.append(primitive)

    def add_library(self, library):
        for p in library._primitives:
            self.add_primitive(p)

    def discard(self, names):
        indices_to_remove = []
        for i, primitive in enumerate(self._primitives):
            if primitive.name in names:
                indices_to_remove.append(i)
        for i in sorted(indices_to_remove, reverse=True):
            del self._primitives[i]

    def name_of(self, i):
        return self._primitives[i].name

    def arity_of(self, i):
        return self._primitives[i].arity

    def parameters_of(self, i):
        return self._primitives[i].n_parameters

    def inputs_of(self, i):
        return self._primitives[i].inputs

    def execute(self, f_index, x: List[np.ndarray], args: List[int]):
        y = self._primitives[f_index].call_f32(x, args)
        return y

    def display(self):
        headers = ["Id", "Name", "Inputs", "Outputs", "Arity", "Parameters"]
        full_list = []
        for i, primitive in enumerate(self._primitives):
            one_primitive_infos = [
                i,
                self.name_of(i),
                self.inputs_of(i),
                primitive.output,
                self.arity_of(i),
                self.parameters_of(i),
            ]
            full_list.append(one_primitive_infos)
        table_name = f"  {self.rtype} Library  "
        print("─" * len(table_name))
        print(table_name)
        print(
            tabulate(
                full_list,
                tablefmt="simple_grid",
                headers=headers,
                numalign="center",
                stralign="center",
            )
        )

    @property
    def random_index(self):
        return random.choice(self.keys)

    @property
    def last_index(self):
        return len(self._primitives) - 1

    @property
    def keys(self):
        return list(range(self.size))

    @property
    def max_arity(self):
        return max([self.arity_of(i) for i in self.keys])

    @property
    def max_parameters(self):
        return max([self.parameters_of(i) for i in self.keys])

    @property
    def size(self):
        return len(self._primitives)


@fundamental()
class Mutation(KartezioComponent, ABC):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.parameters = None  # MutationUniform()
        self.edges_weights = None  # MutationEdgesUniform()

    def random_parameters(self, chromosome: int):
        return np.random.randint(
            self.parameters.max_value,
            size=self.adapter.chromosomes_infos[chromosome].n_parameters,
        )

    def random_function(self, chromosome: str):
        return np.random.randint(
            self.adapter.chromosomes_infos[chromosome].n_functions
        )

    def mutate_function(self, genotype: Genotype, chromosome: str, idx: int):
        self.adapter.set_function(
            genotype, chromosome, idx, self.random_function(chromosome)
        )

    def mutate_edges(
        self,
        genotype: Genotype,
        chromosome: str,
        idx: int,
        only_one: int = None,
    ):
        n_previous_nodes = 1 + idx
        p = self.edges_weights.weights_edges(n_previous_nodes)
        new_edges = np.random.choice(
            list(range(n_previous_nodes)),
            size=self.adapter.chromosomes_infos[chromosome].n_edges,
            p=p,
        )
        for edge, new_edge in enumerate(new_edges):
            if new_edge == 0:
                # sample from inputs
                new_edges[edge] = np.random.randint(self.adapter.n_inputs)
            else:
                # connect to previous nodes
                new_edges[edge] = new_edge - 1 + self.adapter.n_inputs
        if only_one is not None:
            new_value = new_edges[only_one]
            new_edges = self.adapter.get_edges(genotype, chromosome, idx)
            new_edges[only_one] = new_value
        self.adapter.set_edges(genotype, chromosome, idx, new_edges)

    def mutate_parameters(
        self,
        genotype: Genotype,
        chromosome: str,
        idx: int,
        only_one: int = None,
    ):
        new_random_parameters = self.random_parameters(chromosome)
        old_parameters = self.adapter.get_parameters(genotype, chromosome, idx)
        new_parameters = self.parameters.adjust(
            old_parameters, new_random_parameters
        )
        if only_one is not None:
            old_parameters[only_one] = new_parameters[only_one]
            new_parameters = old_parameters.copy()
        self.adapter.set_parameters(genotype, chromosome, idx, new_parameters)

    def mutate_output(self, genotype: Genotype, idx: int):
        n_previous_nodes = 1 + self.adapter.n_nodes
        p = self.edges_weights.weights_edges(n_previous_nodes)
        new_edges = np.random.choice(range(n_previous_nodes), size=1, p=p)
        self.adapter.set_output(genotype, idx, new_edges)
        for edge, new_edge in enumerate(new_edges):
            if new_edge == 0:
                # sample from inputs
                new_edges[edge] = np.random.randint(self.adapter.n_inputs)
            else:
                # connect to previous nodes
                new_edges[edge] = new_edge - 1 + self.adapter.n_inputs

    @abstractmethod
    def mutate(self, genotype: Genotype):
        pass

    def __to_dict__(self) -> Dict:
        return {}


@fundamental()
class Initialization(KartezioComponent, ABC):
    pass
