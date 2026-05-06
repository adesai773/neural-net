from abc import ABC, abstractmethod

import numpy as np

from .node import Node


class Layer(ABC):
    def __call__(self, *args: Node) -> Node:
        return Node(np.array(0))

    def parameters(self) -> list[Node]:
        return []

    @abstractmethod
    def forward(self, *args: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray]:
        pass

    def __str__(self) -> str:
        return "Layer()"


class Linear(Layer):
    def __init__(self) -> None:
        pass

    def forward(self, *args: np.ndarray) -> np.ndarray:
        return np.array(0)

    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray]:
        return []

    def parameters(self) -> list[Node]:
        return []

    def __str__(self) -> str:
        return "Linear()"
