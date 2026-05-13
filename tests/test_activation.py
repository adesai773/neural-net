import numpy as np
import pytest

from neural_net.activation import ReLU


@pytest.fixture
def relu() -> ReLU:
    return ReLU()


def test_relu_call(relu: ReLU):
    assert relu() is not None


def test_relu_forward(relu: ReLU):
    assert relu.forward() is not None


def test_relu_backward(relu: ReLU):
    assert relu.backward(np.array(0), []) is not None
