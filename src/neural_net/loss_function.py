from abc import ABC, abstractmethod

import numpy as np

from .node import Node


class LossFunction(ABC):
    def __call__(self, y_pred: Node, y_true: Node | np.ndarray) -> Node:
        return Node(np.array(0))

    @abstractmethod
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray]:
        pass

    def __str__(self) -> str:
        return "LossFunction()"


class MseLoss(LossFunction):
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return 0

    def backward(
        self, upstream_grad: np.ndarray, parents: list[Node]
    ) -> list[np.ndarray]:
        return []

    def __str__(self) -> str:
        return "MseLoss()"
