import numpy as np

from neural_net.node import Node
from neural_net.utils import topological_sort


def test_topological_sort_root():
    root = Node(np.array(0))

    assert topological_sort(root) == [root]


def test_topological_sort_basic_tree():
    parent_1 = Node(np.array(0))
    parent_2 = Node(np.array(0))
    root = Node(np.array(0), parents=[parent_1, parent_2])

    assert topological_sort(root) == [parent_1, parent_2, root]


def test_topological_sort_tree_3_layers():
    grandparent_1 = Node(np.array(0))
    grandparent_2 = Node(np.array(0))
    parent_1 = Node(np.array(0), parents=[grandparent_1, grandparent_2])
    parent_2 = Node(np.array(0))
    root = Node(np.array(0), parents=[parent_1, parent_2])

    assert topological_sort(root) == [
        grandparent_1,
        grandparent_2,
        parent_1,
        parent_2,
        root,
    ]


def test_topological_sort_root_does_not_require_grad():
    parent_1 = Node(np.array(0))
    parent_2 = Node(np.array(0))
    root = Node(np.array(0), parents=[parent_1, parent_2], requires_grad=False)

    assert topological_sort(root) == []


def test_topological_sort_other_node_does_not_require_grad():
    grandparent_1 = Node(np.array(0))
    grandparent_2 = Node(np.array(0))
    parent_1 = Node(
        np.array(0), parents=[grandparent_1, grandparent_2], requires_grad=False
    )
    parent_2 = Node(np.array(0))
    root = Node(np.array(0), parents=[parent_1, parent_2])

    assert topological_sort(root) == [
        parent_2,
        root,
    ]


def test_topological_sort_basic_diamond():
    common_grandparent = Node(np.array(0))
    parent_1 = Node(np.array(0), parents=[common_grandparent])
    parent_2 = Node(np.array(0), parents=[common_grandparent])
    root = Node(np.array(0), parents=[parent_1, parent_2])

    assert topological_sort(root) == [
        common_grandparent,
        parent_1,
        parent_2,
        root,
    ]


def test_topological_sort_diamond_with_parent():
    great_grandparent = Node(np.array(0))
    common_grandparent = Node(np.array(0), parents=[great_grandparent])
    parent_1 = Node(np.array(0), parents=[common_grandparent])
    parent_2 = Node(np.array(0), parents=[common_grandparent])
    root = Node(np.array(0), parents=[parent_1, parent_2])

    assert topological_sort(root) == [
        great_grandparent,
        common_grandparent,
        parent_1,
        parent_2,
        root,
    ]


def test_topological_sort_diamond_requires_grad_false():
    common_grandparent = Node(np.array(0))
    parent_1 = Node(np.array(0), parents=[common_grandparent], requires_grad=False)
    parent_2 = Node(np.array(0), parents=[common_grandparent])
    root = Node(np.array(0), parents=[parent_1, parent_2])

    assert topological_sort(root) == [
        common_grandparent,
        parent_2,
        root,
    ]


def test_topological_sort_linear_chain():
    node_a = Node(np.array(0))
    node_b = Node(np.array(0), parents=[node_a])
    node_c = Node(np.array(0), parents=[node_b])
    root = Node(np.array(0), parents=[node_c])

    assert topological_sort(root) == [node_a, node_b, node_c, root]


def test_topological_sort_linear_chain_requires_grad_false():
    node_a = Node(np.array(0))
    node_b = Node(np.array(0), parents=[node_a])
    node_c = Node(np.array(0), parents=[node_b], requires_grad=False)
    root = Node(np.array(0), parents=[node_c])

    assert topological_sort(root) == [root]


def test_topological_sort_duplicate_parent():
    parent = Node(np.array(0))
    root = Node(np.array(0), parents=[parent, parent])

    assert topological_sort(root) == [parent, root]


def test_topological_sort_many_parents():
    parents = [Node(np.array(0)) for _ in range(10)]
    root = Node(np.array(0), parents=parents)

    expected = [*parents, root]
    assert topological_sort(root) == expected


def test_topological_sort_mixed_depths():
    node_a = Node(np.array(0))
    node_b = Node(np.array(0), parents=[node_a])
    node_c = Node(np.array(0), parents=[node_b])
    node_1 = Node(np.array(0))
    root = Node(np.array(0), parents=[node_c, node_1])

    assert topological_sort(root) == [node_a, node_b, node_c, node_1, root]
