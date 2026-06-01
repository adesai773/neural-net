import numpy as np
import pytest

from neural_net.layer import Linear
from neural_net.node import Node


@pytest.fixture
def linear() -> Linear:
    return Linear(3, 2)


def test_linear_rng_seed():
    a = Linear(3, 2, seed=42).W_node.data
    b = Linear(3, 2, seed=42).W_node.data
    assert np.array_equal(a, b)


def test_linear_different_seeds_differ():
    a = Linear(3, 2, seed=42).W_node.data
    b = Linear(3, 2, seed=43).W_node.data
    assert not np.array_equal(a, b)


def test_linear_call(linear: Linear):
    assert linear() is not None


def test_linear_forward(linear: Linear):
    linear.W_node.data = np.array([[1, 2], [3, 4], [5, 6]])
    linear.b_node.data = np.array([1, 1])

    assert np.array_equal(linear.forward(np.array([[1, 1, 1]])), np.array([[10, 13]]))


def test_linear_forward_with_batch(linear: Linear):
    linear.W_node.data = np.array([[1, 2], [3, 4], [5, 6]])
    linear.b_node.data = np.array([1, 1])

    assert np.array_equal(
        linear.forward(np.array([[1, 1, 1], [2, 0, 0]])), np.array([[10, 13], [3, 5]])
    )


def test_linear_backward(linear: Linear):
    linear.W_node.data = np.array([[1, 2], [3, 4], [5, 6]])
    linear.b_node.data = np.array([1, 1])

    x_in_node = Node(np.array([[1, 1, 1]]))

    assert linear.W_node.grad is None
    assert linear.b_node.grad is None

    parent_grad = linear.backward(np.array([[1, 2]]), [x_in_node])

    assert len(parent_grad) == 1
    assert np.array_equal(parent_grad[0], np.array([[5, 11, 17]]))
    assert linear.W_node.grad is not None
    assert np.array_equal(linear.W_node.grad, np.array([[1, 2], [1, 2], [1, 2]]))
    assert linear.b_node.grad is not None
    assert np.array_equal(linear.b_node.grad, np.array([1, 2]))


def test_linear_backward_accumulates(linear: Linear):
    linear.W_node.data = np.array([[1, 2], [3, 4], [5, 6]])
    linear.b_node.data = np.array([1, 1])

    x_in_node = Node(np.array([[1, 1, 1]]))
    parent_grad = linear.backward(np.array([[1, 2]]), [x_in_node])
    parent_grad_2 = linear.backward(np.array([[1, 2]]), [x_in_node])

    # Gradients accumulate for W and b, stay the same for parents.
    assert np.array_equal(parent_grad, parent_grad_2)
    assert np.array_equal(parent_grad_2[0], np.array([[5, 11, 17]]))
    assert linear.W_node.grad is not None
    assert np.array_equal(linear.W_node.grad, np.array([[2, 4], [2, 4], [2, 4]]))
    assert linear.b_node.grad is not None
    assert np.array_equal(linear.b_node.grad, np.array([2, 4]))


def test_linear_parameters(linear: Linear):
    assert linear.parameters() == [linear.W_node, linear.b_node]
