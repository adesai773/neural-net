from abc import ABC, abstractmethod

import numpy as np

from .node import Node


class LossFunction(ABC):
    def __call__(self, y_pred: Node, y_true: Node | np.ndarray) -> Node:
        if not isinstance(y_true, Node):
            y_true = Node(y_true, requires_grad=False)

        raw_pred = y_pred.data
        raw_true = y_true.data

        raw_loss = self.compute_loss(raw_pred, raw_true)
        return Node(
            np.array(raw_loss),
            creator=self,
            parents=[y_pred, y_true],
            requires_grad=y_pred.requires_grad,
        )

    @abstractmethod
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray | None]:
        pass

    def __str__(self) -> str:
        return "LossFunction()"


class MseLoss(LossFunction):
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
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
        return "MseLoss()"
