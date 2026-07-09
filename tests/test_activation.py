import numpy as np
import pytest

from neural_net.activation import ReLU
from neural_net.node import Node


@pytest.fixture
def relu() -> ReLU:
    return ReLU()


def test_relu_call(relu: ReLU):
    X_in_node = Node(np.array([[1, 1, 1]]))
    relu_out_node = relu(X_in_node)

    assert relu_out_node is not None
    assert relu_out_node.creator is relu
    assert relu_out_node.parents == [X_in_node]
    assert np.array_equal(relu_out_node.data, relu.forward(X_in_node.data))
    assert relu_out_node.requires_grad


def test_relu_forward(relu: ReLU):
    X_in = np.array([[-1, 0], [1, 2], [-3.5, 1.5]])
    assert np.array_equal(relu.forward(X_in), np.array([[0, 0], [1, 2], [0, 1.5]]))


def test_relu_backward(relu: ReLU):
    X_in = np.array([[-1, 0], [1, -2], [-3.5, 1.5]])
    X_in_node = Node(X_in)

    grads = relu.backward(np.array([[0, 1], [-1, 2], [3.5, -1.5]]), [X_in_node])
    assert len(grads) == 1
    assert grads[0] is not None

    assert np.array_equal(grads[0], np.array([[0, 0], [-1, 0], [0, -1.5]]))
