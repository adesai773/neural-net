import numpy as np
import pytest

from neural_net.node import Node


@pytest.fixture
def node() -> Node:
    return Node(np.array(0))


def test_node_accumulate_grad(node: Node):
    node.accumulate_grad(np.array(1))
    assert node.grad is not None
    assert np.array_equal(node.grad, np.array(1))

    node.accumulate_grad(np.array(1))
    assert node.grad is not None
    assert np.array_equal(node.grad, np.array(2))


def test_node_backward(node: Node):
    assert node.backward() is None
