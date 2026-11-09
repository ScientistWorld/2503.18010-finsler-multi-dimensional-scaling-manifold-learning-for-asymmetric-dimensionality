"""Directed binary-tree hierarchy data from the Finsler MDS paper.

The paper's graph-hierarchy visualisation (Appendix D.1.1) uses a complete
binary tree with cheap parent-to-child edges (0.5), expensive child-to-parent
edges (1.5), and small symmetric links (0.1) between all nodes at the same
depth. Shortest-path distances on this directed weighted graph form the
asymmetric dissimilarity matrix.

Besides the exact paper tree, this module exposes the *hierarchy family* the
gym's train/test slices draw from: the same tree topology with per-level edge
weights drawn around the paper's values and a random node labelling. This
public copy is for method development only; the evaluators keep their own
copies, so editing this file never changes a score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


PAPER_PARENT_TO_CHILD = 0.5
PAPER_CHILD_TO_PARENT = 1.5
PAPER_SAME_DEPTH = 0.1

# Ranges of the per-level hierarchy family (see sample_hierarchy_instance).
FAMILY_SCALE_RANGE = (0.8, 1.2)        # multiplies the 0.5 parent->child weight
FAMILY_UP_DOWN_RATIO_RANGE = (2.5, 3.0)  # child->parent / parent->child weight
FAMILY_SAME_DEPTH_RANGE = (0.6, 1.4)   # multiplies the 0.1 same-depth weight


@dataclass(frozen=True)
class DirectedTreeData:
    adjacency: np.ndarray
    distances: np.ndarray
    depths: np.ndarray


def _per_level(value, depth: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(depth, float(arr))
    if arr.shape != (depth,):
        raise ValueError(f"{name} must be a scalar or have one entry per level 1..{depth}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return arr


def make_directed_binary_tree(
    depth: int = 7,
    parent_to_child=PAPER_PARENT_TO_CHILD,
    child_to_parent=PAPER_CHILD_TO_PARENT,
    same_depth=PAPER_SAME_DEPTH,
) -> DirectedTreeData:
    """Build the directed weighted tree used in Appendix D.1.1.

    Each weight may be a scalar (the paper's setting) or an array with one
    entry per level ``1..depth``: entry ``l-1`` of ``parent_to_child`` and
    ``child_to_parent`` weights the edges between levels ``l-1`` and ``l``,
    and entry ``l-1`` of ``same_depth`` weights the undirected links inside
    level ``l`` (level 0 holds only the root).
    """

    if depth < 1:
        raise ValueError("depth must be at least 1")
    down = _per_level(parent_to_child, depth, "parent_to_child")
    up = _per_level(child_to_parent, depth, "child_to_parent")
    same = _per_level(same_depth, depth, "same_depth")

    n_nodes = 2 ** (depth + 1) - 1
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    depths = np.floor(np.log2(np.arange(n_nodes) + 1)).astype(np.int64)

    for node in range(n_nodes):
        for child in (2 * node + 1, 2 * node + 2):
            if child < n_nodes:
                level = depths[child]
                adjacency[node, child] = down[level - 1]
                adjacency[child, node] = up[level - 1]

    for level in range(1, depth + 1):
        nodes = np.flatnonzero(depths == level)
        rows, cols = np.meshgrid(nodes, nodes, indexing="ij")
        mask = rows != cols
        adjacency[rows[mask], cols[mask]] = same[level - 1]

    distances = shortest_path(csr_matrix(adjacency), directed=True, unweighted=False)
    if not np.isfinite(distances).all():
        raise RuntimeError("directed tree should be strongly connected")
    return DirectedTreeData(adjacency=adjacency, distances=distances, depths=depths)


def sample_hierarchy_instance(depth: int, seed: int) -> DirectedTreeData:
    """Draw one tree from the per-level hierarchy family, with shuffled labels.

    Every level gets its own scale for the vertical edges, its own up/down
    cost ratio and its own same-depth link weight; node labels are randomly
    permuted so node indices carry no information about the hierarchy.
    """

    rng = np.random.default_rng(seed)
    scale = rng.uniform(*FAMILY_SCALE_RANGE, size=depth)
    ratio = rng.uniform(*FAMILY_UP_DOWN_RATIO_RANGE, size=depth)
    same = PAPER_SAME_DEPTH * rng.uniform(*FAMILY_SAME_DEPTH_RANGE, size=depth)
    down = PAPER_PARENT_TO_CHILD * scale
    tree = make_directed_binary_tree(depth, down, ratio * down, same)
    order = rng.permutation(tree.distances.shape[0])
    return DirectedTreeData(
        adjacency=tree.adjacency[np.ix_(order, order)],
        distances=tree.distances[np.ix_(order, order)],
        depths=tree.depths[order],
    )


def save_tree_npz(path: str, tree: DirectedTreeData) -> None:
    np.savez_compressed(
        path,
        adjacency=tree.adjacency,
        distances=tree.distances,
        depths=tree.depths,
    )
