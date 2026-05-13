import numpy as np

from neural_net.node import Node
from neural_net.utils import topological_sort


def test_topological_sort():
    assert topological_sort(Node(np.array(0))) == []
