from abc import ABC, abstractmethod

import numpy as np

from .node import Node


class Optimizer(ABC):
    def __init__(self, parameters: list[Node], learning_rate: float = 0.01) -> None:
        self.parameters: list[Node] = parameters
        self.learning_rate: float = learning_rate

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None

    @abstractmethod
    def step(self) -> None:
        pass

    def __str__(self) -> str:
        return "Optimizer()"


class Sgd(Optimizer):
    def step(self) -> None:
        for param in self.parameters:
            if param.grad is not None and param.requires_grad:
                param.data -= self.learning_rate * param.grad

    def __str__(self) -> str:
        return f"Sgd(learning_rate={self.learning_rate})"


class Momentum(Optimizer):
    def __init__(
        self, parameters: list[Node], learning_rate: float = 0.01, momentum: float = 0.9
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.momentum: float = momentum
        self.velocities: list[np.ndarray] = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is not None and param.requires_grad:
                # Note: I'm using the EMA formulation of momentum here
                # (as opposed to the "classical" heavy-ball formulation)
                self.velocities[i] = (
                    self.momentum * self.velocities[i]
                    + (1 - self.momentum) * param.grad
                )
                param.data -= self.learning_rate * self.velocities[i]

    def __str__(self) -> str:
        return f"Momentum(learning_rate={self.learning_rate}, momentum={self.momentum})"


class RMSprop(Optimizer):
    def __init__(
        self,
        parameters: list[Node],
        learning_rate: float = 0.01,
        decay: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learning_rate)

        self.decay: float = decay
        self.epsilon: float = epsilon

        self.squared_grad_ema: list[np.ndarray] = [
            np.zeros_like(p.data) for p in parameters
        ]

    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is not None and param.requires_grad:
                self.squared_grad_ema[i] = (
                    self.decay * self.squared_grad_ema[i]
                    + (1 - self.decay) * param.grad**2
                )
                param.data -= (
                    self.learning_rate
                    * param.grad
                    / (np.sqrt(self.squared_grad_ema[i]) + self.epsilon)
                )

    def __str__(self) -> str:
        return f"RMSprop(learning_rate={self.learning_rate}, decay={self.decay}, epsilon={self.epsilon})"


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[Node],
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(parameters, learning_rate)

        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.iteration: int = 0

        self.m: list[np.ndarray] = [np.zeros_like(p.data) for p in parameters]
        self.v: list[np.ndarray] = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        self.iteration += 1
        bias1 = 1 - self.beta1**self.iteration
        bias2 = 1 - self.beta2**self.iteration

        for i, param in enumerate(self.parameters):
            if param.grad is not None and param.requires_grad:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad**2

                # Bias correction
                m = self.m[i] / bias1
                v = self.v[i] / bias2

                param.data -= self.learning_rate * m / (np.sqrt(v) + self.epsilon)

    def __str__(self) -> str:
        return f"Adam(learning_rate={self.learning_rate}, beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon})"
