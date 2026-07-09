import numpy as np
import pytest

from neural_net.layer import Linear
from neural_net.loss_function import MseLoss
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
    linear = Linear(3, 2)
    linear.W_node.data = np.array([[1, 2], [3, 4], [5, 6]])
    linear.b_node.data = np.array([1, 1])

    x = Node(np.array([[1, 1, 1]]), requires_grad=False)
    y_true = np.array([[0, 0]])

    y_pred = linear(x)
    loss = MseLoss()(y_pred, y_true)
    loss.backward()

    assert loss.grad is not None
    assert y_pred.grad is not None
    assert linear.W_node.grad is not None
    assert linear.b_node.grad is not None
    assert x.grad is None

    assert np.array_equal(y_pred.grad, np.array([[10, 13]]))
    assert np.array_equal(linear.W_node.grad, np.array([[10, 13], [10, 13], [10, 13]]))
    assert np.array_equal(linear.b_node.grad, np.array([10, 13]))


def test_node_backward_twice_without_zero_grad(node: Node):
    linear = Linear(3, 2)
    linear.W_node.data = np.array([[1, 2], [3, 4], [5, 6]])
    linear.b_node.data = np.array([1, 1])

    x = Node(np.array([[1, 1, 1]]), requires_grad=False)
    y_true = np.array([[0, 0]])

    y_pred = linear(x)
    loss = MseLoss()(y_pred, y_true)
    loss.backward()
    loss.backward()

    assert loss.grad is not None
    assert y_pred.grad is not None
    assert linear.W_node.grad is not None
    assert linear.b_node.grad is not None
    assert x.grad is None

    assert np.array_equal(y_pred.grad, np.array([[20, 26]]))
    assert np.array_equal(linear.W_node.grad, np.array([[30, 39], [30, 39], [30, 39]]))
    assert np.array_equal(linear.b_node.grad, np.array([30, 39]))
