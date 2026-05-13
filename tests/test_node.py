import numpy as np
import pytest

from neural_net.node import Node


@pytest.fixture
def node() -> Node:
    return Node(np.array(0))


def test_node_backward(node: Node):
    assert node.backward() is None
