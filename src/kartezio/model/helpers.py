from abc import ABC, abstractmethod
from typing import List


def singleton(cls):
    """
    https://towardsdatascience.com/10-fabulous-python-decorators-ab674a732871
    """
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


class Prototype(ABC):
    """
    Using Prototype Pattern to duplicate:
    https://refactoring.guru/design-patterns/prototype
    """

    @abstractmethod
    def clone(self):
        pass


class Factory:
    """
    Using Factory Pattern:
    https://refactoring.guru/design-patterns/factory-method
    """

    def __init__(self, prototype):
        self._prototype = None
        self.set_prototype(prototype)

    def set_prototype(self, prototype):
        self._prototype = prototype

    def create(self):
        return self._prototype.clone()


class Observer(ABC):
    """
    The Observer interface declares the update method, used by subjects.
    """

    @abstractmethod
    def update(self, event):
        """
        Receive update from subject.
        """
        pass


class Observable(ABC):
    """
    For the sake of simplicity, the Observable state, essential to all
    subscribers, is stored in this variable.
    """

    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def clear(self) -> None:
        self._observers = []

    def notify(self, event) -> None:
        for observer in self._observers:
            observer.update(event)
