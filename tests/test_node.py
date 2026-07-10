import numpy as np
import pytest

from neural_net.activation import ReLU
from neural_net.layer import Linear
from neural_net.loss_function import MseLoss
from neural_net.node import Node


@pytest.fixture
def node() -> Node:
    return Node(np.array(0))


def test_node_add():
    node_1 = Node(np.array([1, 2]))
    node_2 = Node(np.array([2, 3]))

    sum_node = node_1 + node_2
    assert isinstance(sum_node, Node)
    assert np.array_equal(sum_node.data, np.array([3, 5]))
    assert sum_node.requires_grad


def test_node_add_with_nparray():
    node_1 = Node(np.array([1, 2]), requires_grad=False)

    sum_node = node_1 + np.array([2, 3])
    assert isinstance(sum_node, Node)
    assert np.array_equal(sum_node.data, np.array([3, 5]))
    assert not sum_node.requires_grad


def test_node_radd_with_nparray():
    node_1 = Node(np.array([1, 2]), requires_grad=False)

    sum_node = np.array([2, 3]) + node_1
    assert isinstance(sum_node, Node)
    assert np.array_equal(sum_node.data, np.array([3, 5]))
    assert not sum_node.requires_grad


def test_node_accumulate_grad(node: Node):
    node.accumulate_grad(np.array(1))
    assert node.grad is not None
    assert np.array_equal(node.grad, np.array(1))

    node.accumulate_grad(np.array(1))
    assert node.grad is not None
    assert np.array_equal(node.grad, np.array(2))


def test_node_backward():
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

    assert np.array_equal(loss.grad, np.array(1))
    assert np.array_equal(y_pred.grad, np.array([[10, 13]]))
    assert np.array_equal(linear.W_node.grad, np.array([[10, 13], [10, 13], [10, 13]]))
    assert np.array_equal(linear.b_node.grad, np.array([10, 13]))


def test_node_backward_twice_without_zero_grad():
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

    assert np.array_equal(loss.grad, np.array(1))
    assert np.array_equal(y_pred.grad, np.array([[10, 13]]))
    assert np.array_equal(linear.W_node.grad, np.array([[20, 26], [20, 26], [20, 26]]))
    assert np.array_equal(linear.b_node.grad, np.array([20, 26]))


def test_node_backward_stacked_layers():
    linear1 = Linear(3, 2)
    linear1.W_node.data = np.array([[1, 0], [1, 0], [1, 0]])
    linear1.b_node.data = np.array([1, 1])

    linear2 = Linear(2, 3)
    linear2.W_node.data = np.array([[0, 0, 0], [2, 2, 2]])
    linear2.b_node.data = np.array([1, 1, 1])

    x = Node(np.array([[1, 1, 1]]), requires_grad=False)
    y_true = np.array([[0, 0, 0]])

    out1 = linear1(x)
    y_pred = linear2(out1)
    loss = MseLoss()(y_pred, y_true)

    loss.backward()

    assert loss.grad is not None
    assert np.array_equal(loss.grad, np.array(1))

    assert y_pred.grad is not None
    assert np.array_equal(y_pred.grad, np.array([[2, 2, 2]]))

    assert out1.grad is not None
    assert np.array_equal(out1.grad, np.array([[0, 12]]))

    assert linear2.W_node.grad is not None
    assert np.array_equal(linear2.W_node.grad, np.array([[8, 8, 8], [2, 2, 2]]))

    assert linear2.b_node.grad is not None
    assert np.array_equal(linear2.b_node.grad, np.array([2, 2, 2]))

    assert x.grad is None

    assert linear1.W_node.grad is not None
    assert np.array_equal(linear1.W_node.grad, np.array([[0, 12], [0, 12], [0, 12]]))

    assert linear1.b_node.grad is not None
    assert np.array_equal(linear1.b_node.grad, np.array([0, 12]))


def test_node_backward_with_activation():
    linear1 = Linear(1, 1)
    linear1.W_node.data = np.array([[1]])
    linear1.b_node.data = np.array([-3])

    relu = ReLU()

    linear2 = Linear(1, 1)
    linear2.W_node.data = np.array([[10]])
    linear2.b_node.data = np.array([5])

    x = Node(np.array([[1]]), requires_grad=False)
    y_true = np.array([[7]])

    linear1_out = linear1(x)
    relu_out = relu(linear1_out)
    y_pred = linear2(relu_out)
    loss = MseLoss()(y_pred, y_true)

    loss.backward()

    assert loss.grad is not None
    assert np.array_equal(loss.grad, np.array(1))

    assert y_pred.grad is not None
    assert np.array_equal(y_pred.grad, np.array([[-4]]))

    assert linear2.W_node.grad is not None
    assert np.array_equal(linear2.W_node.grad, np.array([[0]]))

    assert linear2.b_node.grad is not None
    assert np.array_equal(linear2.b_node.grad, np.array([-4]))

    assert relu_out.grad is not None
    assert np.array_equal(relu_out.grad, np.array([[-40]]))

    assert linear1_out.grad is not None
    assert np.array_equal(linear1_out.grad, np.array([[0]]))

    assert linear1.W_node.grad is not None
    assert np.array_equal(linear1.W_node.grad, np.array([[0]]))

    assert linear1.b_node.grad is not None
    assert np.array_equal(linear1.b_node.grad, np.array([0]))

    assert x.grad is None


def test_node_backward_diamond():
    linear1 = Linear(1, 1)
    linear1.W_node.data = np.array([[1]])
    linear1.b_node.data = np.array([1])

    linear2 = Linear(1, 1)
    linear2.W_node.data = np.array([[-2]])
    linear2.b_node.data = np.array([0])

    linear3 = Linear(1, 1)
    linear3.W_node.data = np.array([[1]])
    linear3.b_node.data = np.array([4])

    x = Node(np.array([[1]]), requires_grad=False)
    linear1_out = linear1(x)
    linear2_out = linear2(linear1_out)
    linear3_out = linear3(linear1_out)
    sum_node = linear2_out + linear3_out

    sum_node.backward()

    assert sum_node.grad is not None
    assert np.array_equal(sum_node.grad, np.array([[1]]))

    assert linear2_out.grad is not None
    assert np.array_equal(linear2_out.grad, np.array([[1]]))

    assert linear2.W_node.grad is not None
    assert np.array_equal(linear2.W_node.grad, np.array([[2]]))

    assert linear2.b_node.grad is not None
    assert np.array_equal(linear2.b_node.grad, np.array([1]))

    assert linear3_out.grad is not None
    assert np.array_equal(linear3_out.grad, np.array([[1]]))

    assert linear3.W_node.grad is not None
    assert np.array_equal(linear3.W_node.grad, np.array([[2]]))

    assert linear3.b_node.grad is not None
    assert np.array_equal(linear3.b_node.grad, np.array([1]))

    assert linear1_out.grad is not None
    # upstream gradients for linear2 and linear3 accumulated
    assert np.array_equal(linear1_out.grad, np.array([[-1]]))

    assert linear1.W_node.grad is not None
    assert np.array_equal(linear1.W_node.grad, np.array([[-1]]))

    assert linear1.b_node.grad is not None
    assert np.array_equal(linear1.b_node.grad, np.array([-1]))


def test_node_backward_frozen_parameter():
    linear = Linear(1, 1)
    linear.W_node.data = np.array([[1]])
    linear.W_node.requires_grad = False
    linear.b_node.data = np.array([1])

    x = Node(np.array([[1]]), requires_grad=False)

    y = linear(x)
    y.backward()

    assert y.grad is not None
    assert np.array_equal(y.grad, np.array([[1]]))

    assert linear.W_node.grad is None

    assert linear.b_node.grad is not None
    assert np.array_equal(linear.b_node.grad, np.array([1]))

    assert x.grad is None


def test_node_backward_explicit_gradient_argument():
    linear = Linear(1, 1)
    linear.W_node.data = np.array([[1]])
    linear.b_node.data = np.array([2])

    x = Node(np.array([[3]]), requires_grad=False)

    y = linear(x)
    y.backward(gradient=np.array([[2]]))

    assert y.grad is not None
    assert np.array_equal(y.grad, np.array([[2]]))

    assert linear.W_node.grad is not None
    assert np.array_equal(linear.W_node.grad, np.array([[6]]))

    assert linear.b_node.grad is not None
    assert np.array_equal(linear.b_node.grad, np.array([2]))

    assert x.grad is None


def test_numerical_gradient_check():
    pass
