from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .loss_function import LossFunction
from .node import Node


class Model(ABC):
    def __init__(self):
        pass

    def __call__(self, *args: Node | np.ndarray) -> list[Node]:
        return []

    def parameters(self) -> list[Node]:
        return []

    def train(
        self,
        x_train: np.ndarray | list[np.ndarray],
        y_true: np.ndarray | list[np.ndarray],
        loss: LossFunction | list[LossFunction],
        optimizer_key: str = "sgd",
        num_epochs: int = 5,
        batch_size: int | None = None,
        learning_rate: float = 0.01,
    ) -> None:
        pass

    def predict(self, *args: np.ndarray) -> list[np.ndarray]:
        return []

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> list[Node]:
        raise NotImplementedError("Users must subclass Model and define graph.")

    def __str__(self) -> str:
        return "Model()"
