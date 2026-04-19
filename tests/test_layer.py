import pytest

from neural_net.layer import LinearLayer


@pytest.fixture
def linear_layer() -> LinearLayer:
    return LinearLayer()


def test_linear_layer_forward(linear_layer: LinearLayer):
    assert linear_layer.forward() is None
    assert True
