from abc import ABC, abstractmethod

from kartezio.model.population import Population


class Strategy(ABC):
    @abstractmethod
    def selection(self, population: Population):
        pass

    @abstractmethod
    def reproduction(self, population: Population):
        pass
