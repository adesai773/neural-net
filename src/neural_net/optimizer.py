from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass

    def __str__(self) -> str:
        return "Optimizer()"


class SgdOptimizer(Optimizer):
    def step(self):
        pass

    def __str__(self) -> str:
        return "SgdOptimizer()"
