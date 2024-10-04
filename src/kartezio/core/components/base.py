import abc
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Dict

from kartezio.core.helpers import Observer


class Component(ABC):
    def __init__(self):
        self.name = Components.name_of(self.__class__)

    def __to_dict__(self) -> Dict:
        return {}

    @classmethod
    @abstractmethod
    def __from_dict__(cls, dict_infos: Dict) -> "Component":
        pass


class UpdatableComponent(Component, Observer, ABC):
    def __init__(self):
        super().__init__()


class Components:
    _registry = {}
    _reverse = {}

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
        assert issubclass(
            component, Component
        ), f"{component} is not a Component!"

        if group_name not in Components._registry.keys():
            Components._registry[group_name] = {}

        if component_name not in Components._registry[group_name].keys():
            Components._registry[group_name][component_name] = component

        Components._registry[group_name][component_name] = component
        Components._reverse[component.__name__] = (
            f"{group_name}/{component_name}"
        )

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
            # print(f"Component '{component_class.__name__}'
            # not properly registered, please make sure use
            # '@register' over your Class definition.")
            return component_class.__name__
        return Components._reverse[component_class.__name__].split("/")[1]

    @staticmethod
    def display():
        pprint(Components._registry)


class Node(Component, ABC):
    @abstractmethod
    def call(self, *args, **kwargs):
        pass


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
        return item_cls

    return inner


def load_component(component_class, json_data: dict) -> Component:
    return component_class.__from_dict__(json_data)
