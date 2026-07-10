import numpy as np

from .layer import Layer
from .node import Node


class Activation(Layer):
    def __str__(self) -> str:
        return "Activation()"


class ReLU(Activation):
    def forward(self, X_in: np.ndarray) -> np.ndarray:
        return np.maximum(X_in, 0)

    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray | None]:
        assert len(parents) == 1
        X = parents[0].data
        assert X.shape == upstream_grad.shape

        downstream_grad = upstream_grad.copy()
        downstream_grad[X <= 0] = 0

        return [downstream_grad]

    def __str__(self) -> str:
        return "ReLU()"
