"""Topological scheduler for DAG execution."""

from typing import Dict, List, Set

import networkx as nx


def topological_sort(nodes: List[str], edges: List[tuple[str, str]]) -> List[str]:
    """
    Perform topological sort on nodes and edges.

    Args:
        nodes: List of node names
        edges: List of (source, target) tuples

    Returns:
        Sorted list of node names in execution order

    Raises:
        ValueError: If graph contains cycles
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph contains cycles - cannot perform topological sort")

    return list(nx.topological_sort(graph))


def get_execution_order(edges: List[tuple[str, str]]) -> List[List[str]]:
    """
    Get execution order as levels (nodes that can run in parallel).

    Args:
        edges: List of (source, target) tuples

    Returns:
        List of levels, where each level contains nodes that can run in parallel
    """
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph contains cycles")

    # Compute levels
    levels = []
    remaining = set(graph.nodes())
    in_degree = dict(graph.in_degree())

    while remaining:
        # Find all nodes with no incoming edges in remaining set
        level = [node for node in remaining if in_degree.get(node, 0) == 0]
        if not level:
            # Cycle detected
            raise ValueError("Graph contains cycles")

        levels.append(level)
        remaining -= set(level)

        # Update in-degrees
        for node in level:
            for successor in graph.successors(node):
                in_degree[successor] = in_degree.get(successor, 0) - 1

    return levels


def get_upstream_nodes(node: str, edges: List[tuple[str, str]]) -> Set[str]:
    """
    Get all upstream nodes (ancestors) of a given node.

    Args:
        node: Target node name
        edges: List of (source, target) tuples

    Returns:
        Set of upstream node names
    """
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    if node not in graph:
        return set()

    # Get all ancestors
    ancestors = set()
    queue = [node]
    visited = {node}

    while queue:
        current = queue.pop(0)
        for predecessor in graph.predecessors(current):
            if predecessor not in visited:
                visited.add(predecessor)
                ancestors.add(predecessor)
                queue.append(predecessor)

    return ancestors


def get_downstream_nodes(node: str, edges: List[tuple[str, str]]) -> Set[str]:
    """
    Get all downstream nodes (descendants) of a given node.

    Args:
        node: Source node name
        edges: List of (source, target) tuples

    Returns:
        Set of downstream node names
    """
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    if node not in graph:
        return set()

    # Get all descendants using DFS
    descendants = set()
    stack = [node]
    visited = {node}

    while stack:
        current = stack.pop()
        for successor in graph.successors(current):
            if successor not in visited:
                visited.add(successor)
                descendants.add(successor)
                stack.append(successor)

    return descendants

