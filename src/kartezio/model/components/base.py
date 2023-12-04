from kartezio.model.helpers import singleton


class Component:
    """
    def _register(self, _class, component: type, name: str):
        Components.register(_class, component, name, replace=False)
        Components.display()

    def __init_subclass__(cls, component: type, name: str = None):
        if name is not None:
            cls._register(cls, component, name)
    """

    pass


class Components:
    _registry = {}

    @classmethod
    def contains(cls, component_type, component_name):
        return (
            component_name in cls._registry[component_type].__components.keys()
        )

    @classmethod
    def add(cls, component_type, component_name, component):
        cls._registry[component_type].__components[
            component_name
        ] = component

    @classmethod
    def register(cls, _class, component, name, replace: bool):
        assert isinstance(_class, type), f"{_class} is not a Class!"
        assert issubclass(_class, Component), f"{_class} is not a Component!"
        class_name = _class.__name__
        if class_name not in cls._registry.keys():
            cls._registry[class_name] = {}

        if name in cls._registry[class_name].keys():
            if not replace:
                raise KeyError(
                    f"Error registering {class_name} called '{name}'."
                    f"Here is the list of all registered {class_name} components:"
                    f"{cls._registry[class_name].keys()}"
                )
        cls._registry[component][name] = _class


@singleton
class Registry:
    class SubRegistry:
        def __init__(self):
            self.__components = {}

        def remove(self):
            pass

        def get(self, item_name):
            if item_name not in self.__components.keys():
                raise ValueError(f"Component '{item_name}' not found in the registry!")
            return self.__components[item_name]

        def instantiate(self, item_name, *args, **kwargs):
            return self.get(item_name)(*args, **kwargs)

        def list(self):
            return self.__components

    def __init__(self):
        self.__registries = {}
        self.__registries[Endpoint] = self.SubRegistry()


def register(component_type, component_name, replace=False):
    def inner(item_cls):
        if Components.contains(component_type, component_name):
            if replace:

            else:
                print(
                    f"Warning, '{component_name}' already registered, replace it using 'replace=True', or use another name."
                )
        else:
            Components._registry[component_type].__components[component_name] = item_cls

        def wrapper(*args, **kwargs):
            return item_cls(*args, **kwargs)

        return wrapper

    return inner


class Endpoint:
    pass


@register(Endpoint, "test")
class Test(object):
    pass
