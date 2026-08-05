import numpy as np
import pytest

from neural_net.node import Node
from neural_net.optimizer import Momentum, RMSprop, Sgd


@pytest.fixture
def sgd_optimizer() -> Sgd:
    param1 = Node(np.array(0))
    param2 = Node(np.array(0))
    return Sgd([param1, param2])


@pytest.fixture
def momentum_optimizer() -> Momentum:
    param1 = Node(np.array(0))
    param2 = Node(np.array(0))
    return Momentum([param1, param2])


@pytest.fixture
def rmsprop_optimizer() -> RMSprop:
    param1 = Node(np.array(0))
    param2 = Node(np.array(0))
    return RMSprop([param1, param2])


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


def test_momentum_zero_grad(momentum_optimizer: Momentum):
    for param in momentum_optimizer.parameters:
        param.grad = np.array(5)

    momentum_optimizer.zero_grad()

    for param in momentum_optimizer.parameters:
        assert param.grad is None


def test_momentum_step(momentum_optimizer: Momentum):
    momentum_optimizer.learning_rate = 1
    momentum_optimizer.momentum = 0.5
    for param in momentum_optimizer.parameters:
        param.data = np.array(10.0)
        param.grad = np.array(4.0)

    momentum_optimizer.step()

    for idx, param in enumerate(momentum_optimizer.parameters):
        assert momentum_optimizer.velocities[idx] == 2.0
        assert np.array_equal(param.data, np.array(8))


def test_momentum_step_skips_when_grad_is_none(momentum_optimizer: Momentum):
    momentum_optimizer.learning_rate = 1
    momentum_optimizer.momentum = 0.5
    assert len(momentum_optimizer.parameters) == 2
    param1 = momentum_optimizer.parameters[0]
    param1.data = np.array(10.0)
    param1.grad = None
    param1.requires_grad = True
    param2 = momentum_optimizer.parameters[1]
    param2.data = np.array(10.0)
    param2.grad = np.array(4.0)
    param2.requires_grad = True

    momentum_optimizer.step()

    assert np.array_equal(param1.data, np.array(10))  # unchanged
    assert np.array_equal(param2.data, np.array(8))


def test_momentum_step_skips_frozen_parameters(momentum_optimizer: Momentum):
    momentum_optimizer.learning_rate = 1
    momentum_optimizer.momentum = 0.5
    assert len(momentum_optimizer.parameters) == 2
    param1 = momentum_optimizer.parameters[0]
    param1.data = np.array(10.0)
    param1.grad = np.array(4.0)
    param1.requires_grad = False
    param2 = momentum_optimizer.parameters[1]
    param2.data = np.array(10.0)
    param2.grad = np.array(4.0)
    param2.requires_grad = True

    momentum_optimizer.step()

    assert np.array_equal(param1.data, np.array(10))  # unchanged
    assert np.array_equal(param2.data, np.array(8))


def test_momentum_velocity_accumulates(momentum_optimizer: Momentum):
    momentum_optimizer.learning_rate = 1
    momentum_optimizer.momentum = 0.5
    for param in momentum_optimizer.parameters:
        param.data = np.array(10.0)
        param.grad = np.array(4.0)

    momentum_optimizer.step()

    for idx, param in enumerate(momentum_optimizer.parameters):
        np.testing.assert_allclose(momentum_optimizer.velocities[idx], 2.0)
        assert np.array_equal(param.data, np.array(8))

    momentum_optimizer.step()

    for idx, param in enumerate(momentum_optimizer.parameters):
        np.testing.assert_allclose(momentum_optimizer.velocities[idx], 3.0)
        assert np.array_equal(param.data, np.array(5))


def test_rmsprop_zero_grad(rmsprop_optimizer: RMSprop):
    for param in rmsprop_optimizer.parameters:
        param.grad = np.array(5)

    rmsprop_optimizer.zero_grad()

    for param in rmsprop_optimizer.parameters:
        assert param.grad is None


def test_rmsprop_step(rmsprop_optimizer: RMSprop):
    rmsprop_optimizer.learning_rate = 1
    rmsprop_optimizer.decay = 0.5
    rmsprop_optimizer.epsilon = 0
    for param in rmsprop_optimizer.parameters:
        param.data = np.array(10.0)
        param.grad = np.array(4.0)

    rmsprop_optimizer.step()

    for idx, param in enumerate(rmsprop_optimizer.parameters):
        assert rmsprop_optimizer.squared_grad_ema[idx] == 8.0
        np.testing.assert_allclose(param.data, 8.586, atol=0.001)


def test_rmsprop_step_skips_when_grad_is_none(rmsprop_optimizer: RMSprop):
    rmsprop_optimizer.learning_rate = 1
    rmsprop_optimizer.decay = 0.5
    rmsprop_optimizer.epsilon = 0
    assert len(rmsprop_optimizer.parameters) == 2
    param1 = rmsprop_optimizer.parameters[0]
    param1.data = np.array(10.0)
    param1.grad = None
    param1.requires_grad = True
    param2 = rmsprop_optimizer.parameters[1]
    param2.data = np.array(10.0)
    param2.grad = np.array(4.0)
    param2.requires_grad = True

    rmsprop_optimizer.step()

    assert np.array_equal(param1.data, np.array(10))  # unchanged
    np.testing.assert_allclose(param2.data, 8.586, atol=0.001)


def test_rmsprop_step_skips_frozen_parameters(rmsprop_optimizer: RMSprop):
    rmsprop_optimizer.learning_rate = 1
    rmsprop_optimizer.decay = 0.5
    rmsprop_optimizer.epsilon = 0
    assert len(rmsprop_optimizer.parameters) == 2
    param1 = rmsprop_optimizer.parameters[0]
    param1.data = np.array(10.0)
    param1.grad = np.array(4.0)
    param1.requires_grad = False
    param2 = rmsprop_optimizer.parameters[1]
    param2.data = np.array(10.0)
    param2.grad = np.array(4.0)
    param2.requires_grad = True

    rmsprop_optimizer.step()

    assert np.array_equal(param1.data, np.array(10))  # unchanged
    np.testing.assert_allclose(param2.data, 8.586, atol=0.001)


def test_rmsprop_state_accumulates(rmsprop_optimizer: RMSprop):
    rmsprop_optimizer.learning_rate = 1
    rmsprop_optimizer.decay = 0.5
    rmsprop_optimizer.epsilon = 0
    for param in rmsprop_optimizer.parameters:
        param.data = np.array(10.0)
        param.grad = np.array(4.0)

    rmsprop_optimizer.step()

    for idx, param in enumerate(rmsprop_optimizer.parameters):
        np.testing.assert_allclose(rmsprop_optimizer.squared_grad_ema[idx], 8.0)
        np.testing.assert_allclose(param.data, 8.586, atol=0.001)

    rmsprop_optimizer.step()

    for idx, param in enumerate(rmsprop_optimizer.parameters):
        np.testing.assert_allclose(rmsprop_optimizer.squared_grad_ema[idx], 12.0)
        np.testing.assert_allclose(param.data, 7.431, atol=0.001)


def test_rmsprop_normalizes_across_grad_magnitudes(rmsprop_optimizer: RMSprop):
    rmsprop_optimizer.learning_rate = 1
    rmsprop_optimizer.decay = 0.5
    rmsprop_optimizer.epsilon = 0
    for i, param in enumerate(rmsprop_optimizer.parameters):
        param.data = np.array(10.0)
        param.grad = np.array(0.01) if i == 0 else np.array(100)

    for _ in range(10):
        rmsprop_optimizer.step()

    param1 = rmsprop_optimizer.parameters[0]
    s1 = rmsprop_optimizer.squared_grad_ema[0]
    param2 = rmsprop_optimizer.parameters[1]
    s2 = rmsprop_optimizer.squared_grad_ema[1]
    np.testing.assert_allclose(
        param1.grad / np.sqrt(s1),
        param2.grad / np.sqrt(s2),
        atol=0.00000001,
    )
