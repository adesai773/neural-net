from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .layer import Layer


class Node:
    def __init__(
        self,
        data: np.ndarray,
        creator: Layer | None = None,
        parents: list[Node] | None = None,
        requires_grad: bool = True,
    ) -> None:
        self.data: np.ndarray = data
        self.creator: Layer | None = creator
        self.parents: list[Node] = parents if parents is not None else []
        self.requires_grad: bool = requires_grad
        self.grad: np.ndarray | None = None

    def accumulate_grad(self, dg: np.ndarray) -> None:
        if self.grad is None:
            self.grad = dg
        else:
            self.grad += dg

    def backward(self) -> None:
        pass

    def __str__(self) -> str:
        return f"Node(data={self.data}, creator={self.creator}, requires_grad={self.requires_grad}, grad={self.grad})"
