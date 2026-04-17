from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
