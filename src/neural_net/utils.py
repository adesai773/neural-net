from .node import Node


def topological_sort(root: Node) -> list[Node]:
    if not root.requires_grad:
        return []

    visited: set[Node] = set()
    order: list[Node] = []

    def dfs(node: Node) -> None:
        if node in visited or not node.requires_grad:
            return

        visited.add(node)
        for parent in node.parents:
            dfs(parent)

        order.append(node)

    dfs(root)
    return order
