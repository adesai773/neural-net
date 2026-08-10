from abc import ABC, abstractmethod

import numpy as np

from .node import Node


class LossFunction(ABC):
    def __call__(self, y_pred: Node, y_true: Node | np.ndarray) -> Node:
        if not isinstance(y_true, Node):
            y_true = Node(y_true, requires_grad=False)

        raw_pred = y_pred.data
        raw_true = y_true.data

        raw_loss = self.forward(raw_pred, raw_true)
        return Node(
            np.array(raw_loss),
            creator=self,
            parents=[y_pred, y_true],
            requires_grad=y_pred.requires_grad,
        )

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray | None]:
        pass

    def __str__(self) -> str:
        return "LossFunction()"


class Mse(LossFunction):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        assert y_pred.shape == y_true.shape, (
            f"y_pred shape {y_pred.shape} needs to match y_true shape {y_true.shape}"
        )
        assert y_pred.ndim == 2, f"y_pred must be 2-d, actual: {y_pred.ndim}"

        N = y_pred.shape[0]
        D = y_pred.shape[1]
        return np.sum((y_pred - y_true) ** 2) / (N * D)

    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray | None]:

        assert len(parents) == 2
        y_pred = parents[0].data
        y_true = parents[1].data

        assert y_pred.ndim == 2
        assert y_true.ndim == 2
        assert y_pred.shape == y_true.shape

        assert upstream_grad.ndim == 0
        assert upstream_grad.size == 1

        N = y_pred.shape[0]
        D = y_pred.shape[1]
        dL_dy_pred = upstream_grad * (2 / (N * D)) * (y_pred - y_true)

        return [dL_dy_pred, None]

    def __str__(self) -> str:
        return "Mse()"


class BceWithLogits(LossFunction):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        assert y_pred.shape == y_true.shape, (
            f"y_pred shape {y_pred.shape} needs to match y_true shape {y_true.shape}"
        )
        assert y_pred.ndim == 2, f"y_pred must be 2-d, actual: {y_pred.ndim}"

        loss_per_elem = (
            np.maximum(y_pred, 0) - y_pred * y_true + np.logaddexp(0, -np.abs(y_pred))
        )
        return float(np.mean(loss_per_elem))

    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray | None]:
        assert len(parents) == 2
        y_pred = parents[0].data
        y_true = parents[1].data

        assert y_pred.ndim == 2
        assert y_true.ndim == 2
        assert y_pred.shape == y_true.shape

        assert upstream_grad.ndim == 0
        assert upstream_grad.size == 1

        N = y_pred.shape[0]
        D = y_pred.shape[1]
        # Note: dividing by (N * D) in case of multi-output BCE
        dL_dy_pred = upstream_grad * (self.sigmoid(y_pred) - y_true) / (N * D)

        return [dL_dy_pred, None]

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            # Stable form (i.e. so exp(z) doesn't overflow)
            return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
