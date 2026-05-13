import numpy as np
import pytest

from neural_net.loss_function import MseLoss
from neural_net.node import Node


@pytest.fixture
def mse_loss() -> MseLoss:
    return MseLoss()


def test_mse_loss_call(mse_loss: MseLoss):
    assert mse_loss(Node(np.array(0)), Node(np.array(0))) is not None


def test_mse_loss_compute_loss(mse_loss: MseLoss):
    assert mse_loss.compute_loss(np.array(0), np.array(0)) == 0


def test_mse_loss_backward(mse_loss: MseLoss):
    assert mse_loss.backward(np.array(0), []) is not None
