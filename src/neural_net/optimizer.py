from abc import ABC, abstractmethod

from .node import Node


class Optimizer(ABC):
    def __init__(self, parameters: list[Node], learning_rate: float = 0.01) -> None:
        self.parameters: list[Node] = parameters
        self.learning_rate: float = learning_rate

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None

    @abstractmethod
    def step(self) -> None:
        pass

    def __str__(self) -> str:
        return "Optimizer()"


class Sgd(Optimizer):
    def step(self) -> None:
        for param in self.parameters:
            if param.grad is not None and param.requires_grad:
                param.data -= self.learning_rate * param.grad

    def __str__(self) -> str:
        return "Sgd()"
