"""Directed citation-graph instances (the paper's Table 1 datasets).

The paper reconstructs directed shortest-path distances of real digraphs
(Cora, Citeseer, ...) in two dimensions and compares Finsler and Euclidean
embeddings by distortion and MAP (Table 1). A citation graph is almost acyclic,
so most directed shortest paths do not exist; following the paper's own
directed-tree construction (cheap parent -> child edges, three times dearer
child -> parent edges), every citation can be followed at cost 1 and travelled
against at cost ``REVERSE_COST`` = 3. Directed shortest paths on that weighted
digraph give a finite, asymmetric dissimilarity matrix.

``data/citation_graphs/*_lwcc_edges.txt`` hold the largest weakly connected
component of each LINQS citation graph (built from the raw files fetched by
``scripts/download.sh`` with ``data/build_citation_graphs.py``). This module is
a public copy of the instance generator for development: the evaluators keep
their own copies of the graphs and of this code, so editing anything here never
changes a score.

    from data.citation_digraph import sample_citation_instance, paper_citation_instance
    inst = sample_citation_instance("cora", 3000)   # a visible train instance
    inst.distances, inst.edges
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

GRAPH_DIR = Path(__file__).resolve().parent / "citation_graphs"
GRAPHS = ("cora", "citeseer")
REVERSE_COST = 3.0
SUBGRAPH_NODES = 400
# Seeds of the visible (train) instances; the test slice draws hidden ones.
TRAIN_SEEDS = {"cora": list(range(3000, 3005)), "citeseer": list(range(4000, 4005))}


@dataclass(frozen=True)
class CitationInstance:
    distances: np.ndarray  # (N, N) directed shortest-path distances, row = source
    edges: np.ndarray      # (N, N) bool, edges[i, j] = paper i cites paper j


@lru_cache(maxsize=None)
def load_graph(name: str):
    """Directed (citing -> cited) and undirected CSR adjacency of the LWCC."""

    lines = [
        line
        for line in (GRAPH_DIR / f"{name}_lwcc_edges.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    n = int(lines[0].split()[1])
    edges = np.array([[int(v) for v in line.split()] for line in lines[1:]], dtype=np.int64)
    directed = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    directed.sum_duplicates()
    directed.data[:] = 1.0
    undirected = (directed + directed.T).tocsr()
    undirected.sum_duplicates()
    undirected.sort_indices()
    return directed, undirected


def bfs_nodes(undirected, start: int, size: int) -> np.ndarray:
    """The first ``size`` nodes reached by breadth-first search (citations in either direction)."""

    seen = np.zeros(undirected.shape[0], dtype=bool)
    seen[start] = True
    order = [int(start)]
    head = 0
    while head < len(order) and len(order) < size:
        u = order[head]
        head += 1
        for v in undirected.indices[undirected.indptr[u] : undirected.indptr[u + 1]]:
            if not seen[v]:
                seen[v] = True
                order.append(int(v))
                if len(order) == size:
                    break
    return np.array(order, dtype=np.int64)


def directed_distances(edges: np.ndarray, reverse_cost: float = REVERSE_COST) -> np.ndarray:
    """Shortest paths with cost 1 along a citation and ``reverse_cost`` against it."""

    n = edges.shape[0]
    src, dst = np.nonzero(edges)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    vals = np.concatenate([np.ones(src.size), np.full(src.size, reverse_cost)])
    key = rows * n + cols
    order = np.lexsort((vals, key))
    first = np.ones(order.size, dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    sel = order[first]
    graph = csr_matrix((vals[sel], (rows[sel], cols[sel])), shape=(n, n))
    distances = shortest_path(graph, method="D", directed=True)
    if not np.isfinite(distances).all():
        raise RuntimeError("instance is not connected")
    return distances


def sample_citation_instance(name: str, seed: int, size: int = SUBGRAPH_NODES) -> CitationInstance:
    """An induced BFS subgraph of ``size`` papers with shuffled node labels."""

    directed, undirected = load_graph(name)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(directed.shape[0]))
    nodes = bfs_nodes(undirected, start, size)
    nodes = nodes[rng.permutation(nodes.size)]
    edges = directed[nodes][:, nodes].toarray() > 0
    return CitationInstance(distances=directed_distances(edges), edges=edges)


def paper_citation_instance(name: str) -> CitationInstance:
    """The whole largest weakly connected component (the paper-setting instance)."""

    directed, _ = load_graph(name)
    edges = directed.toarray() > 0
    return CitationInstance(distances=directed_distances(edges), edges=edges)
