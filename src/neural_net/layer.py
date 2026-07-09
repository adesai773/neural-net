from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .node import Node


class Layer(ABC):
    def __call__(self, *args: Node) -> Node:
        raw_data = [node.data for node in args]
        raw_params = [p.data for p in self.parameters()]

        out_data = self.forward(*raw_data, *raw_params)
        requires_grad = any(node.requires_grad for node in args) or any(
            p.requires_grad for p in self.parameters()
        )

        return Node(
            out_data,
            creator=self,
            parents=[*args, *self.parameters()],
            requires_grad=requires_grad,
        )

    def parameters(self) -> list[Node]:
        return []

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> np.ndarray:
        pass

    @abstractmethod
    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray | None]:
        pass

    def __str__(self) -> str:
        return "Layer()"


class Add(Layer):
    def __init__(self) -> None:
        pass

    def forward(self, *node_data: np.ndarray) -> np.ndarray:
        assert node_data, "Need to have nonzero number of node_data"

        unique_shapes = {data.shape for data in node_data}
        assert len(unique_shapes) == 1, "All node_data must have the same shape"

        return np.sum(node_data, axis=0)

    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray | None]:
        # Y = (I) x A + (I) x B
        # dL/dA = dY/dA x dL/dY = I.T * dL/dY = I x upstream_grad = upstream_grad

        assert len(parents) > 0
        unique_shapes = {parent.data.shape for parent in parents}
        assert len(unique_shapes) == 1, "All node_data must have the same shape"
        assert upstream_grad.shape == parents[0].data.shape

        return [upstream_grad] * len(parents)


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

    def forward(self, X_in: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        assert X_in.ndim == 2, (
            f"Linear.forward expects 2-D input (batch, in_features), got shape {X_in.shape}"
        )
        assert W.ndim == 2, (
            f"Linear.forward expects 2-D Weights (in_features, out_features), got shape {W.shape}"
        )
        assert b.ndim == 1, (
            f"Linear.forward expects 1-D biases (out_features), got shape {b.shape}"
        )
        in_features = W.shape[0]
        out_features = W.shape[1]

        assert X_in.shape[1] == in_features
        assert b.shape[0] == out_features

        # X_in dimensions: N x in_features
        # X_in @ W produces dimensions N x out_features
        # adding to b_node (broadcast to N x out_features) produces dimensions N x out_features
        return X_in @ W + b

    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray | None]:
        assert len(parents) == 3, (
            f"Linear.backward expects exactly 3 parents, got {len(parents)}"
        )

        X = parents[0].data
        W = parents[1].data
        b = parents[2].data

        assert X.ndim == 2, (
            f"Linear.backward expects 2-D parent (input) data, got shape {X.shape}"
        )
        assert W.ndim == 2, (
            f"Linear.backward expects 2-D parent (weights) data, got shape {W.shape}"
        )
        assert b.ndim == 1, (
            f"Linear.backward expects 1-D parent (biases), got shape {b.shape}"
        )

        expected_upstream_grad_shape = (X.shape[0], W.shape[1])
        assert upstream_grad.shape == expected_upstream_grad_shape, (
            f"upstream_grad must be {expected_upstream_grad_shape}, got {upstream_grad.shape}"
        )

        dL_dY = upstream_grad  # dimensions: N x out_features
        dL_dX = dL_dY @ W.T  # dimensions N x in_features
        dL_dW = X.T @ dL_dY  # dimensions in_features x out_features
        dL_db = dL_dY.sum(axis=0)  # dimensions out_features

        return [dL_dX, dL_dW, dL_db]

    def parameters(self) -> list[Node]:
        return [self.W_node, self.b_node]

    def __str__(self) -> str:
        return "Linear()"
