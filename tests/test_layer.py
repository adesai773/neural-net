import numpy as np
import pytest

from neural_net.layer import Add, Linear
from neural_net.node import Node


@pytest.fixture
def linear() -> Linear:
    return Linear(3, 2)


@pytest.fixture
def add() -> Add:
    return Add()


def test_add_call(add: Add):
    A_node = Node(np.array([1, 2, 3]))
    B_node = Node(np.array([-1, 0, 1]))

    Y_node = add(A_node, B_node)
    assert Y_node is not None
    assert Y_node.creator is add
    assert Y_node.parents == [A_node, B_node]
    assert np.array_equal(
        Y_node.data,
        add.forward(A_node.data, B_node.data),
    )
    assert Y_node.requires_grad


def test_add_forward(add: Add):
    A = np.array([1, 2, 3])
    B = np.array([0, 0, 1])
    C = np.array([1, 0, 0])

    assert np.array_equal(add.forward(A, B, C), np.array([2, 2, 4]))


def test_add_backward(add: Add):
    A = np.array([1, 2, 3])
    B = np.array([0, 0, 1])
    C = np.array([1, 0, 0])

    parents = [Node(A), Node(B), Node(C)]
    grads = add.backward(upstream_grad=np.array([10, 20, 40]), parents=parents)

    assert len(grads) == 3
    assert grads[0] is not None
    assert np.array_equal(grads[0], np.array([10, 20, 40]))
    assert grads[1] is not None
    assert np.array_equal(grads[1], np.array([10, 20, 40]))
    assert grads[2] is not None
    assert np.array_equal(grads[2], np.array([10, 20, 40]))


def test_linear_rng_seed():
    a = Linear(3, 2, seed=42).W_node.data
    b = Linear(3, 2, seed=42).W_node.data
    assert np.array_equal(a, b)


def test_linear_different_seeds_differ():
    a = Linear(3, 2, seed=42).W_node.data
    b = Linear(3, 2, seed=43).W_node.data
    assert not np.array_equal(a, b)


def test_linear_call(linear: Linear):
    linear.W_node.data = np.array([[1, 2], [3, 4], [5, 6]])
    linear.b_node.data = np.array([1, 1])

    X_in_node = Node(np.array([[1, 1, 1]]))

    out_node = linear(X_in_node)
    assert out_node is not None
    assert out_node.creator is linear
    assert out_node.parents == [X_in_node, linear.W_node, linear.b_node]
    assert np.array_equal(
        out_node.data,
        linear.forward(X_in_node.data, linear.W_node.data, linear.b_node.data),
    )
    assert out_node.requires_grad


def test_linear_call_requires_grad_propagation(linear: Linear):
    X_in_node = Node(np.array([[1, 1, 1]]), requires_grad=False)

    out_node = linear(X_in_node)
    assert out_node is not None
    assert out_node.requires_grad


def test_linear_forward(linear: Linear):
    linear.W_node.data = np.array([[1, 2], [3, 4], [5, 6]])
    linear.b_node.data = np.array([1, 1])

    assert np.array_equal(
        linear.forward(np.array([[1, 1, 1]]), linear.W_node.data, linear.b_node.data),
        np.array([[10, 13]]),
    )


def test_linear_forward_with_batch(linear: Linear):
    linear.W_node.data = np.array([[1, 2], [3, 4], [5, 6]])
    linear.b_node.data = np.array([1, 1])

    assert np.array_equal(
        linear.forward(
            np.array([[1, 1, 1], [2, 0, 0]]), linear.W_node.data, linear.b_node.data
        ),
        np.array([[10, 13], [3, 5]]),
    )


def test_linear_backward(linear: Linear):
    linear.W_node.data = np.array([[1, 2], [3, 4], [5, 6]])
    linear.b_node.data = np.array([1, 1])

    X_in_node = Node(np.array([[1, 1, 1]]))
    parents = [X_in_node, linear.W_node, linear.b_node]

    grads = linear.backward(np.array([[1, 2]]), parents)
    dL_dX, dL_dW, dL_db = grads

    assert np.array_equal(dL_dX, np.array([[5, 11, 17]]))
    assert np.array_equal(dL_dW, np.array([[1, 2], [1, 2], [1, 2]]))
    assert np.array_equal(dL_db, np.array([1, 2]))


def test_linear_parameters(linear: Linear):
    assert linear.parameters() == [linear.W_node, linear.b_node]
