import abc
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Dict

from kartezio.model.helpers import Observer


class Component(ABC):
    def __to_dict__(self) -> Dict:
        return {}

    @classmethod
    @abstractmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Component":
        pass


class UpdatableComponent(Component, Observer, ABC):
    def __init__(self, name):
        super().__init__(name)


class Components:
    _registry = {}

    @staticmethod
    def _contains(group_name: str, component_name: str):
        if group_name not in Components._registry.keys():
            return False
        if component_name not in Components._registry[group_name].keys():
            return False
        return True

    @staticmethod
    def contains(component_type: type, component_name: str):
        return Components._contains(component_type.__name__, component_name)

    @staticmethod
    def add(group_name: str, component_name: str, component: type):
        assert isinstance(component, type), f"{component} is not a Class!"
        assert issubclass(component, Component), f"{component} is not a Component!"

        if group_name not in Components._registry.keys():
            Components._registry[group_name] = {}

        if component_name not in Components._registry[group_name].keys():
            Components._registry[group_name][component_name] = component

        Components._registry[group_name][component_name] = component

    @staticmethod
    def instantiate(group_name: str, component_name: str, *args, **kwargs):
        if not Components._contains(group_name, component_name):
            raise KeyError(
                f"Component '{group_name}', called '{component_name}' not found in the registry!"
            )
        return Components._registry[group_name][component_name](*args, **kwargs)

    @staticmethod
    def display():
        pprint(Components._registry)


class Node(Component, ABC, component="Node"):
    def __init__(self, fn: Callable, **kwargs):
        assert callable(
            fn
        ), f"given 'function' {fn.__name__} is not callable! (type: {type(fn)})"
        super().__init__()
        self._fn = fn
        self.__kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def register(component_group: type, component_name: str, replace=False):
    group_name = component_group.__name__

    def inner(item_cls):
        if Components._contains(group_name, component_name):
            if not replace:
                raise KeyError(
                    f"""Error registering {group_name} called '{component_name}'.
                    Here is the list of all registered {group_name} components:
                    \n{Components._registry[group_name].keys()}.
                    \n > Replace it using 'replace=True' in @register, or use another name.
                """
                )
        Components.add(group_name, component_name, item_cls)

        def wrapper(*args, **kwargs):
            return item_cls(*args, **kwargs)

        return wrapper

    return inner
