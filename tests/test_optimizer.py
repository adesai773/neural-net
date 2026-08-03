import numpy as np
import pytest

from neural_net.node import Node
from neural_net.optimizer import Sgd


@pytest.fixture
def sgd_optimizer() -> Sgd:
    param1 = Node(np.array(0))
    param2 = Node(np.array(0))
    return Sgd([param1, param2])


def test_sgd_zero_grad(sgd_optimizer: Sgd):
    for param in sgd_optimizer.parameters:
        param.grad = np.array(5)

    sgd_optimizer.zero_grad()

    for param in sgd_optimizer.parameters:
        assert param.grad is None


def test_sgd_step(sgd_optimizer: Sgd):
    sgd_optimizer.learning_rate = 2
    for param in sgd_optimizer.parameters:
        param.data = np.array(10)
        param.grad = np.array(4)

    sgd_optimizer.step()

    for param in sgd_optimizer.parameters:
        assert np.array_equal(param.data, np.array(2))


def test_sgd_step_skips_when_grad_is_none(sgd_optimizer: Sgd):
    sgd_optimizer.learning_rate = 1
    assert len(sgd_optimizer.parameters) == 2
    param1 = sgd_optimizer.parameters[0]
    param1.data = np.array(10)
    param1.grad = None
    param1.requires_grad = True
    param2 = sgd_optimizer.parameters[1]
    param2.data = np.array(10)
    param2.grad = np.array(4)
    param2.requires_grad = True

    sgd_optimizer.step()

    assert np.array_equal(param1.data, np.array(10))  # unchanged
    assert np.array_equal(param2.data, np.array(6))


def test_sgd_step_skips_frozen_parameters(sgd_optimizer: Sgd):
    sgd_optimizer.learning_rate = 1
    assert len(sgd_optimizer.parameters) == 2
    param1 = sgd_optimizer.parameters[0]
    param1.data = np.array(10)
    param1.grad = np.array(4)
    param1.requires_grad = False
    param2 = sgd_optimizer.parameters[1]
    param2.data = np.array(10)
    param2.grad = np.array(4)
    param2.requires_grad = True

    sgd_optimizer.step()

    assert np.array_equal(param1.data, np.array(10))  # unchanged
    assert np.array_equal(param2.data, np.array(6))
