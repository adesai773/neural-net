import numpy as np
import pytest

from neural_net.layer import Linear


@pytest.fixture
def linear() -> Linear:
    return Linear()


def test_linear_call(linear: Linear):
    assert linear() is not None


def test_linear_forward(linear: Linear):
    assert linear.forward() is not None


def test_linear_backward(linear: Linear):
    assert linear.backward(np.array(0), []) is not None


def test_linear_parameters(linear: Linear):
    assert linear.parameters() is not None
