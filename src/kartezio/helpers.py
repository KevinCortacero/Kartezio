from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Type, TypeVar

T = TypeVar("T")


def singleton(cls: Type[T]) -> Callable[..., T]:
    """
    Singleton decorator to ensure a class has only one instance.

    Reference:
    https://towardsdatascience.com/10-fabulous-python-decorators-ab674a732871

    Args:
        cls (Type[T]): The class to apply the singleton pattern to.

    Returns:
        Callable[..., T]: A wrapper function that returns the single instance of the class.
    """
    instances: Dict[Type[T], T] = {}

    def wrapper(*args, **kwargs) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


class Prototype(ABC):
    """
    Abstract base class for using the Prototype Pattern to duplicate objects.

    The Prototype Pattern is used to create new objects by copying an existing object, known as the prototype.
    Reference:
    https://refactoring.guru/design-patterns/prototype
    """

    @abstractmethod
    def clone(self) -> T:
        """
        Create a copy of the current instance.

        Returns:
            T: A new instance that is a copy of the current instance.
        """
        pass


class Factory:
    """
    A Factory class for creating objects using the Factory Pattern.

    The Factory Pattern provides a method to create objects without specifying the exact class of object that will be created.
    Reference:
    https://refactoring.guru/design-patterns/factory-method
    """

    def __init__(self, prototype: Prototype):
        """
        Initialize the Factory with a prototype object.

        Args:
            prototype (Prototype): The prototype instance used to create new objects.
        """
        self._prototype = None
        self.set_prototype(prototype)

    def set_prototype(self, prototype: Prototype) -> None:
        """
        Set the prototype for the factory.

        Args:
            prototype (Prototype): The prototype instance to be used by the factory.
        """
        self._prototype = prototype

    def create(self) -> Prototype:
        """
        Create a new instance by cloning the prototype.

        Returns:
            Prototype: A new instance created from the prototype.
        """
        return self._prototype.clone()


class Observer(ABC):
    """
    Abstract base class for the Observer interface, declaring the `update` method used by subjects to notify observers.
    """

    @abstractmethod
    def update(self, event: str) -> None:
        """
        Receive an update from the subject.

        Args:
            event (str): The event or message being communicated to the observer.
        """
        raise NotImplementedError()


class Observable(ABC):
    """
    Abstract base class for an Observable, allowing other objects to subscribe and receive notifications.

    Manages a list of observers, notifying them of any changes or events.
    """

    def __init__(self):
        """
        Initialize an Observable instance with an empty list of observers.
        """
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        """
        Attach an observer to the observable.

        Args:
            observer (Observer): The observer to attach.
        """
        assert isinstance(observer, Observer), (
            "Observer must be an instance of the Observer class."
        )
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        """
        Detach an observer from the observable.

        Args:
            observer (Observer): The observer to detach.
        """
        self._observers.remove(observer)

    def clear(self) -> None:
        """
        Clear all observers from the observable.
        """
        self._observers = []

    def notify(self, event: str) -> None:
        """
        Notify all attached observers about an event.

        Args:
            event (str): The event or message to notify observers about.
        """
        for observer in self._observers:
            observer.update(event)
