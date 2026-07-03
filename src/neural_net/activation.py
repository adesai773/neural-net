import numpy as np

from .layer import Layer
from .node import Node


class Activation(Layer):
    def __str__(self) -> str:
        return "Activation()"


class ReLU(Activation):
    def __init__(self) -> None:
        pass

    def forward(self, *args: np.ndarray) -> np.ndarray:
        return np.array(0)

    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray | None]:
        return []

    def __str__(self) -> str:
        return "ReLU()"
