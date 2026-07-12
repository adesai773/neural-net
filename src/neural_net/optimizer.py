from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .node import Node


class Optimizer(ABC):
    REGISTRY: ClassVar[dict[str, type[Optimizer]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Automatically register any subclass by its name (e.g., Sgd -> sgd)
        name = cls.__name__.lower()
        Optimizer.REGISTRY[name] = cls

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
