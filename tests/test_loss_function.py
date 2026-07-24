import numpy as np
import pytest

from neural_net.loss_function import Mse
from neural_net.node import Node


@pytest.fixture
def mse_loss() -> Mse:
    return Mse()


def test_mse_loss_call(mse_loss: Mse):
    y_pred = np.array([[2, 2], [2, 4], [1, 1]])
    y_pred_node = Node(y_pred)
    y_true = np.zeros_like(y_pred)

    loss_node = mse_loss(y_pred_node, y_true)
    assert loss_node is not None
    assert loss_node.creator is mse_loss
    assert len(loss_node.parents) == 2
    assert loss_node.parents[0] == y_pred_node
    assert np.array_equal(loss_node.parents[1].data, y_true)
    assert loss_node.data == mse_loss.compute_loss(y_pred, y_true)
    assert loss_node.requires_grad


def test_mse_loss_call_with_true_node(mse_loss: Mse):
    y_pred = np.array([[2, 2], [2, 4], [1, 1]])
    y_pred_node = Node(y_pred)
    y_true = np.zeros_like(y_pred)
    y_true_node = Node(y_true)

    loss_node = mse_loss(y_pred_node, y_true_node)
    assert loss_node is not None
    assert loss_node.creator is mse_loss
    assert loss_node.parents == [y_pred_node, y_true_node]
    assert loss_node.data == mse_loss.compute_loss(y_pred, y_true)
    assert loss_node.requires_grad


def test_mse_loss_call_with_require_grad_false(mse_loss: Mse):
    y_pred = np.array([[2, 2], [2, 4], [1, 1]])
    y_pred_node = Node(y_pred, requires_grad=False)
    y_true = np.zeros_like(y_pred)
    y_true_node = Node(y_true)

    loss_node = mse_loss(y_pred_node, y_true_node)
    assert loss_node is not None
    assert loss_node.creator is mse_loss
    assert loss_node.parents == [y_pred_node, y_true_node]
    assert loss_node.data == mse_loss.compute_loss(y_pred, y_true)
    assert not loss_node.requires_grad


def test_mse_loss_compute_loss(mse_loss: Mse):
    y_pred = np.array([[2, 2], [2, 4], [1, 1]])
    y_true = np.zeros_like(y_pred)

    assert mse_loss.compute_loss(y_pred, y_true) == 5


def test_mse_loss_compute_loss_nonzero_y_true(mse_loss: Mse):
    y_pred = np.array([[4, 3], [9, 100], [1, 4]])
    y_true = np.array([[2, 1], [7, 104], [2, 5]])

    assert mse_loss.compute_loss(y_pred, y_true) == 5


def test_mse_loss_backward(mse_loss: Mse):
    y_pred = np.array([[24, 24], [12, 48], [12, 12]])
    y_pred_node = Node(y_pred)
    y_true = np.zeros_like(y_pred)
    y_true_node = Node(y_true)

    upstream_grad = np.array(1)

    downstream_grads = mse_loss.backward(upstream_grad, [y_pred_node, y_true_node])
    assert len(downstream_grads) == 2
    assert downstream_grads[0] is not None
    assert np.array_equal(downstream_grads[0], np.array([[8, 8], [4, 16], [4, 4]]))
    assert downstream_grads[1] is None


def test_mse_loss_backward_with_upstream_grad(mse_loss: Mse):
    y_pred = np.array([[24, 24], [12, 48], [12, 12]])
    y_pred_node = Node(y_pred)
    y_true = np.zeros_like(y_pred)
    y_true_node = Node(y_true)

    upstream_grad = np.array(0.5)

    downstream_grads = mse_loss.backward(upstream_grad, [y_pred_node, y_true_node])
    assert len(downstream_grads) == 2
    assert downstream_grads[0] is not None
    assert np.array_equal(downstream_grads[0], np.array([[4, 4], [2, 8], [2, 2]]))
    assert downstream_grads[1] is None


def test_mse_loss_backward_with_nonzero_y_true(mse_loss: Mse):
    y_pred = np.array([[24, 24], [12, 48], [12, 12]])
    y_pred_node = Node(y_pred)
    y_true = np.array([[24, 24], [12, 48], [6, 9]])
    y_true_node = Node(y_true)

    upstream_grad = np.array(1)

    downstream_grads = mse_loss.backward(upstream_grad, [y_pred_node, y_true_node])
    assert len(downstream_grads) == 2
    assert downstream_grads[0] is not None
    assert np.array_equal(downstream_grads[0], np.array([[0, 0], [0, 0], [2, 1]]))
    assert downstream_grads[1] is None
