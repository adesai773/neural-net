from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .node import Node


class Optimizer(ABC):
    REGISTRY: ClassVar[dict[str, type[Optimizer]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Automatically register any subclass by its name (e.g., SgdOptimizer -> sgd)
        name = cls.__name__.lower().replace("optimizer", "")
        Optimizer.REGISTRY[name] = cls

    def __init__(self, parameters: list[Node], learning_rate: float = 0.01) -> None:
        self.parameters: list[Node] = parameters
        self.learning_rate: float = learning_rate

    def zero_grad(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    def __str__(self) -> str:
        return "Optimizer()"


class SgdOptimizer(Optimizer):
    def step(self) -> None:
        pass

    def __str__(self) -> str:
        return "SgdOptimizer()"
