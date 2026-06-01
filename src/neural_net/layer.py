from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .node import Node


class Layer(ABC):
    def __call__(self, *args: Node) -> Node:
        return Node(np.array(0))

    def parameters(self) -> list[Node]:
        return []

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> np.ndarray:
        pass

    @abstractmethod
    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray]:
        pass

    def __str__(self) -> str:
        return "Layer()"


class Linear(Layer):
    def __init__(
        self, in_features: int, out_features: int, seed: int | None = None
    ) -> None:
        self._rng = np.random.default_rng(seed)

        # dimensions: in_features x out_features
        scale = np.sqrt(2 / in_features)  # He initialization scaling factor
        self.W_node: Node = Node(
            data=self._rng.standard_normal((in_features, out_features)) * scale
        )
        # dimensions: out_features
        self.b_node: Node = Node(data=np.zeros(out_features))

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        assert X_in.ndim == 2, (
            f"Linear.forward expects 2-D input (batch, in_features), got shape {X_in.shape}"
        )

        # X_in dimensions: N x in_features
        # X_in @ self.W_node.data produces dimensions N x out_features
        # adding to b_node (broadcast to N x out_features) produces dimensions N x out_features
        return X_in @ self.W_node.data + self.b_node.data

    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray]:
        assert len(parents) == 1, (
            f"Linear.backward expects exactly one parent, got {len(parents)}"
        )

        # Only 1 input node (X_in) with dimensions N x in_features
        X = parents[0].data
        assert X.ndim == 2, (
            f"Linear.backward expects 2-D parent data, got shape {X.shape}"
        )

        expected_upstream_grad_shape = (X.shape[0], self.W_node.data.shape[1])
        assert upstream_grad.shape == expected_upstream_grad_shape, (
            f"upstream_grad must be {expected_upstream_grad_shape}, got {upstream_grad.shape}"
        )

        dL_dY = upstream_grad  # dimensions: N x out_features
        dL_dX = dL_dY @ self.W_node.data.T  # dimensions N x in_features
        dL_dW = X.T @ dL_dY  # dimensions in_features x out_features
        dL_db = dL_dY.sum(axis=0)  # dimensions out_features

        self.W_node.accumulate_grad(dL_dW)
        self.b_node.accumulate_grad(dL_db)

        # Single output (dL / dX_in) to match the single parent
        return [dL_dX]  # Dimensions N x in_features

    def parameters(self) -> list[Node]:
        return [self.W_node, self.b_node]

    def __str__(self) -> str:
        return "Linear()"
