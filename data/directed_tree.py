"""Directed binary-tree benchmark data from the Finsler MDS paper.

The paper's graph-hierarchy visualisation uses a complete binary tree with
cheap parent-to-child edges, expensive child-to-parent edges, and small
symmetric links between nodes at the same depth. Shortest-path distances on
this directed weighted graph form the asymmetric dissimilarity matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


@dataclass(frozen=True)
class DirectedTreeData:
    adjacency: np.ndarray
    distances: np.ndarray
    depths: np.ndarray


def make_directed_binary_tree(
    depth: int = 7,
    parent_to_child: float = 0.5,
    child_to_parent: float = 1.5,
    same_depth: float = 0.1,
) -> DirectedTreeData:
    """Build the directed weighted tree used in Appendix D.1.1."""

    if depth < 1:
        raise ValueError("depth must be at least 1")
    n_nodes = 2 ** (depth + 1) - 1
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    depths = np.zeros(n_nodes, dtype=np.int64)

    for node in range(n_nodes):
        level = int(np.floor(np.log2(node + 1)))
        depths[node] = level
        left = 2 * node + 1
        right = 2 * node + 2
        for child in (left, right):
            if child < n_nodes:
                adjacency[node, child] = parent_to_child
                adjacency[child, node] = child_to_parent

    for level in range(depth + 1):
        nodes = np.flatnonzero(depths == level)
        if nodes.size <= 1:
            continue
        rows, cols = np.meshgrid(nodes, nodes, indexing="ij")
        mask = rows != cols
        adjacency[rows[mask], cols[mask]] = same_depth

    distances = shortest_path(csr_matrix(adjacency), directed=True, unweighted=False)
    if not np.isfinite(distances).all():
        raise RuntimeError("directed tree should be strongly connected")
    return DirectedTreeData(adjacency=adjacency, distances=distances, depths=depths)


def save_tree_npz(path: str, tree: DirectedTreeData) -> None:
    np.savez_compressed(
        path,
        adjacency=tree.adjacency,
        distances=tree.distances,
        depths=tree.depths,
    )
