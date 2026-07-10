from typing import TYPE_CHECKING

import numpy as np

from .utils import topological_sort

if TYPE_CHECKING:
    from .layer import Layer
    from .loss_function import LossFunction


class Node:
    # To prevent incorrect decomposition of numpy arrays for radd
    __array_ufunc__ = None

    def __init__(
        self,
        data: np.ndarray,
        creator: Layer | LossFunction | None = None,
        parents: list[Node] | None = None,
        requires_grad: bool = True,
    ) -> None:
        self.data: np.ndarray = data
        self.creator: Layer | LossFunction | None = creator
        self.parents: list[Node] = parents if parents is not None else []
        self.requires_grad: bool = requires_grad
        self.grad: np.ndarray | None = None

    def __add__(self, other: Node | np.ndarray) -> Node:
        if not isinstance(other, Node):
            other = Node(other, requires_grad=False)

        from .layer import Add

        return Add()(self, other)

    def __radd__(self, other: Node | np.ndarray) -> Node:
        return self.__add__(other)

    def accumulate_grad(self, dg: np.ndarray) -> None:
        if self.grad is None:
            self.grad = dg
        else:
            self.grad += dg

    def backward(self, gradient: np.ndarray | None = None) -> None:
        # Calling .backward() twice without zeroing grads first will accumulate
        # gradients on leaves. Use Optimizer.zero_grad() between training steps.

        assert self.requires_grad
        if gradient is None:
            assert self.data.size == 1, "Default gradient only works for scalar outputs"
            gradient = np.ones_like(self.data)
        else:
            assert gradient.shape == self.data.shape
        self.grad = gradient

        topologically_sorted_nodes = topological_sort(self)

        # Reset non-leaf nodes' grads to avoid compounding across backward calls
        # (i.e. if you didn't call Optimizer.zero_grad() between calls)
        for node in topologically_sorted_nodes:
            if node.creator is not None and node is not self:
                node.grad = None

        for node in reversed(topologically_sorted_nodes):
            if node.creator is None or node.grad is None:
                continue

            downstream_grads = node.creator.backward(
                upstream_grad=node.grad, parents=node.parents
            )
            assert len(downstream_grads) == len(node.parents)

            for parent_grad, parent in zip(downstream_grads, node.parents):
                if not parent.requires_grad or parent_grad is None:
                    continue

                parent.accumulate_grad(parent_grad)

    def __str__(self) -> str:
        return f"Node(data={self.data}, creator={self.creator}, requires_grad={self.requires_grad}, grad={self.grad})"
