import numpy as np

from .layer import Layer
from .loss_function import LossFunction
from .optimizer import Optimizer


class NeuralNetwork:
    def __init__(self, optimizer: Optimizer, loss_fn: LossFunction):
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._layers: list[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        self._layers.append(layer)

    def train(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 4, batch_size: int = 8
    ) -> None:
        pass

    def predict(self, x: np.ndarray) -> None:
        pass

    def __str__(self) -> str:
        layers_str = " -> ".join(str(layer) for layer in self._layers)
        return f"NeuralNetwork({layers_str}, {self._optimizer}, {self._loss_fn})"
