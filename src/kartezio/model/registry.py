from kartezio.model.helpers import singleton


@singleton
class Registry:
    class SubRegistry:
        def __init__(self):
            self.__components = {}

        def remove(self):
            pass

        def add(self, item_name, replace=False):
            def inner(item_cls):
                if item_name in self.__components.keys():
                    if replace:
                        self.__components[item_name] = item_cls
                    else:
                        print(
                            f"Warning, '{item_name}' already registered, replace it using 'replace=True', or use another name."
                        )
                else:
                    self.__components[item_name] = item_cls

                def wrapper(*args, **kwargs):
                    return item_cls(*args, **kwargs)

                return wrapper

            return inner

        def get(self, item_name):
            if item_name not in self.__components.keys():
                raise ValueError(f"Component '{item_name}' not found in the registry!")
            return self.__components[item_name]

        def instantiate(self, item_name, *args, **kwargs):
            return self.get(item_name)(*args, **kwargs)

        def list(self):
            return self.__components

    def __init__(self):
        self.primitives = self.SubRegistry()
        self.stackers = self.SubRegistry()
        self.endpoints = self.SubRegistry()
        self.fitness = self.SubRegistry()
        self.metrics = self.SubRegistry()
        self.mutations = self.SubRegistry()
        self.readers = self.SubRegistry()


registry = Registry()
