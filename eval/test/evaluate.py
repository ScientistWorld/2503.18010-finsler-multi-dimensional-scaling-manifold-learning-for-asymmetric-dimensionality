"""Held-out (test) slice evaluator for the directed-distance embedding gym.

Runs every method registered on disk (``outputs/**/method.json``) on hidden
instances and scores the embeddings it returns (2D for the trees, 10D for
the citation graphs; ``index.json`` gives the dimension of every instance):

* directed binary-tree hierarchies drawn from the per-level hierarchy family
  (``directed_tree_reconstruction``: depth 7, ``directed_tree_depth5_reconstruction``:
  depth 5), and
* induced subgraphs of the paper's real citation digraphs
  (``cora_digraph_reconstruction``, ``citeseer_digraph_reconstruction``), where
  following a citation costs 1 and travelling against it costs ``REVERSE_COST``.

The instance seeds live only in this file; the graphs are this evaluator's own
copies (checked by SHA-256). Nothing a method writes is trusted except its
coordinates and its drift vector; every distance and every metric is
recomputed here, and the frozen Euclidean SMACOF baseline is always scored
under the name ``euclidean_mds``.

The method interface, metrics, eligibility gates and failure values are the
same as on the visible slice: a method that returns an invalid output for any
instance, misses its experiment's gate on any instance, or does not beat the
empty prediction is scored as failed (every metric of every experiment
replaced by the value of the empty prediction).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


SLICE = "test"
WORKSPACE = Path(__file__).resolve().parents[2]
CARD_ROOT = WORKSPACE / "outputs"
GRAPH_DIR = Path(__file__).resolve().parent / "graphs"
# Embedding dimension per instance kind (index.json gives it per instance).
EMBEDDING_DIMS = {"tree": 2, "graph": 10}
# Hang guard only. The reference needs a small fraction of this, so its scores
# do not depend on machine load.
METHOD_TIMEOUT_SECONDS = 4 * 3600
BASELINE_NAME = "euclidean_mds"
MAP_K = 10

TREE_EXPERIMENTS = {
    "directed_tree_reconstruction": {"depth": 7, "tag": "d7"},
    "directed_tree_depth5_reconstruction": {"depth": 5, "tag": "d5"},
}
GRAPH_EXPERIMENTS = {
    "cora_digraph_reconstruction": {"graph": "cora", "tag": "cora"},
    "citeseer_digraph_reconstruction": {"graph": "citeseer", "tag": "citeseer"},
}
# Hidden instance streams. The test slice is larger than the visible one to
# keep the measurement noise of paired comparisons small. The tree stream is
# unchanged from the earlier hardening; the citation subgraphs have their own.
HIDDEN_ENTROPY = 0x6A1F93C257D0B8E41C36D2A9
GRAPH_HIDDEN_ENTROPY = 0x3E9B51D7A4C2F08619B7E5D3
N_INSTANCES = {
    "directed_tree_reconstruction": 12,
    "directed_tree_depth5_reconstruction": 20,
    "cora_digraph_reconstruction": 10,
    "citeseer_digraph_reconstruction": 10,
}
METHOD_SEED_BASE = {
    "directed_tree_reconstruction": 0,
    "directed_tree_depth5_reconstruction": 100,
    "cora_digraph_reconstruction": 200,
    "citeseer_digraph_reconstruction": 300,
}

# Per-level hierarchy family around the paper's weights (0.5, 1.5, 0.1).
PAPER_DOWN, PAPER_UP, PAPER_SAME = 0.5, 1.5, 0.1
SCALE_RANGE = (0.8, 1.2)
UP_DOWN_RATIO_RANGE = (2.5, 3.0)
SAME_DEPTH_RANGE = (0.6, 1.4)

# Citation digraphs: processed LINQS edge lists shipped with this evaluator.
GRAPH_FILES = {
    "cora": ("cora_lwcc_edges.txt", "c8de7426f2636911aa3d3dcf615094774e0d39e8ad95b90fbe9e4d7e0c3a3370"),
    "citeseer": ("citeseer_lwcc_edges.txt", "944ea8d68e68012b208cf60750e83ffeb4d52b456088b502bfaa5e2aaa748a32"),
}
REVERSE_COST = 3.0  # the paper's tree uses up/down = 1.5/0.5 = 3
SUBGRAPH_NODES = 400

CLAIM_THRESHOLDS = {
    "distance_preservation_pass": ("normalized_stress", 0.10, "le"),
    "asymmetry_preservation_pass": ("asymmetry_rmse", 0.10, "le"),
    "hierarchy_direction_pass": ("hierarchy_alignment", 0.80, "ge"),
}
# Every instance must satisfy its experiment's gate for the method to be eligible.
TREE_GATE = {
    "normalized_stress": (0.10, "le"),
    "distortion": (0.10, "le"),
    "asymmetry_rmse": (0.10, "le"),
    "hierarchy_alignment": (0.80, "ge"),
}
GRAPH_GATE = {
    "normalized_stress": (0.25, "le"),
    "asymmetry_error": (0.80, "le"),
}
TREE_RAW_METRICS = (
    "normalized_stress",
    "distortion",
    "map_at_10",
    "asymmetry_rmse",
    "asymmetry_correlation",
    "symmetric_rmse",
    "hierarchy_alignment",
)
GRAPH_RAW_METRICS = (
    "normalized_stress",
    "distortion",
    "map",
    "asymmetry_error",
    "symmetric_error",
)


class InvalidOutput(Exception):
    """A method output that cannot be scored."""


# --------------------------------------------------------------------------
# Tree instances
# --------------------------------------------------------------------------


def _tree_distances(depth, down, up, same):
    n = 2 ** (depth + 1) - 1
    levels = np.floor(np.log2(np.arange(n) + 1)).astype(np.int64)
    adjacency = np.zeros((n, n), dtype=np.float64)
    for node in range(n):
        for child in (2 * node + 1, 2 * node + 2):
            if child < n:
                adjacency[node, child] = down[levels[child] - 1]
                adjacency[child, node] = up[levels[child] - 1]
    for level in range(1, depth + 1):
        nodes = np.flatnonzero(levels == level)
        rows, cols = np.meshgrid(nodes, nodes, indexing="ij")
        mask = rows != cols
        adjacency[rows[mask], cols[mask]] = same[level - 1]
    distances = shortest_path(csr_matrix(adjacency), directed=True, unweighted=False)
    if not np.isfinite(distances).all():
        raise RuntimeError("benchmark tree is not strongly connected")
    return distances, levels


def _family_tree(depth, seed):
    rng = np.random.default_rng(seed)
    scale = rng.uniform(*SCALE_RANGE, size=depth)
    ratio = rng.uniform(*UP_DOWN_RATIO_RANGE, size=depth)
    same = PAPER_SAME * rng.uniform(*SAME_DEPTH_RANGE, size=depth)
    down = PAPER_DOWN * scale
    distances, levels = _tree_distances(depth, down, ratio * down, same)
    order = rng.permutation(distances.shape[0])
    return distances[np.ix_(order, order)], levels[order]


# --------------------------------------------------------------------------
# Citation-digraph instances
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _citation_graph(name: str):
    """Directed (citing -> cited) and undirected adjacency of the graph's LWCC."""

    filename, expected = GRAPH_FILES[name]
    path = GRAPH_DIR / filename
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"{path} does not match the benchmark graph (sha256 mismatch)")
    lines = [line for line in raw.decode("utf-8").splitlines() if line and not line.startswith("#")]
    n = int(lines[0].split()[1])
    edges = np.array([[int(v) for v in line.split()] for line in lines[1:]], dtype=np.int64)
    directed = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    directed.sum_duplicates()
    directed.data[:] = 1.0
    undirected = (directed + directed.T).tocsr()
    undirected.sum_duplicates()
    undirected.sort_indices()
    return directed, undirected


def _bfs_nodes(undirected, start: int, size: int) -> np.ndarray:
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


def _digraph_distances(edges: np.ndarray) -> np.ndarray:
    """Directed shortest paths: cost 1 along a citation, REVERSE_COST against it."""

    n = edges.shape[0]
    src, dst = np.nonzero(edges)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    vals = np.concatenate([np.ones(src.size), np.full(src.size, REVERSE_COST)])
    key = rows * n + cols
    order = np.lexsort((vals, key))
    first = np.ones(order.size, dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    sel = order[first]
    graph = csr_matrix((vals[sel], (rows[sel], cols[sel])), shape=(n, n))
    distances = shortest_path(graph, method="D", directed=True)
    if not np.isfinite(distances).all():
        raise RuntimeError("citation instance is not connected")
    return distances


def _citation_instance(name: str, seed: int):
    directed, undirected = _citation_graph(name)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(directed.shape[0]))
    nodes = _bfs_nodes(undirected, start, SUBGRAPH_NODES)
    nodes = nodes[rng.permutation(nodes.size)]
    edges = directed[nodes][:, nodes].toarray() > 0
    return _digraph_distances(edges), edges


def _hidden_seeds(entropy: int, experiments: dict) -> dict[str, list[int]]:
    total = sum(N_INSTANCES[experiment] for experiment in experiments)
    state = np.random.SeedSequence(entropy).generate_state(total, dtype=np.uint32)
    seeds: dict[str, list[int]] = {}
    offset = 0
    for experiment in experiments:
        count = N_INSTANCES[experiment]
        seeds[experiment] = [int(s) for s in state[offset : offset + count]]
        offset += count
    return seeds


def build_instances() -> dict[str, list[dict]]:
    instances: dict[str, list[dict]] = {}
    tree_seeds = _hidden_seeds(HIDDEN_ENTROPY, TREE_EXPERIMENTS)
    for experiment, spec in TREE_EXPERIMENTS.items():
        items = []
        for k, seed in enumerate(tree_seeds[experiment]):
            distances, levels = _family_tree(spec["depth"], seed)
            items.append(
                {"id": f"{spec['tag']}_{k:02d}", "kind": "tree", "distances": distances,
                 "depths": levels, "method_seed": METHOD_SEED_BASE[experiment] + k}
            )
        instances[experiment] = items
    graph_seeds = _hidden_seeds(GRAPH_HIDDEN_ENTROPY, GRAPH_EXPERIMENTS)
    for experiment, spec in GRAPH_EXPERIMENTS.items():
        items = []
        for k, seed in enumerate(graph_seeds[experiment]):
            distances, edges = _citation_instance(spec["graph"], seed)
            items.append(
                {"id": f"{spec['tag']}_{k:02d}", "kind": "graph", "distances": distances,
                 "edges": edges, "method_seed": METHOD_SEED_BASE[experiment] + k}
            )
        instances[experiment] = items
    return instances


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def randers_distances(embedding: np.ndarray, drift: np.ndarray) -> np.ndarray:
    delta = embedding[None, :, :] - embedding[:, None, :]  # delta[i, j] = x_j - x_i
    return np.linalg.norm(delta, axis=-1) + delta @ drift


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def map_at_k(target: np.ndarray, predicted: np.ndarray, k: int = MAP_K) -> float:
    """Tie-aware MAP of the k nearest directed neighbours.

    Every node tied with the k-th nearest target distance counts as relevant;
    exact ties in the predicted ranking are broken pessimistically.
    """

    n = target.shape[0]
    k = min(k, n - 1)
    aps = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        t = target[i, keep]
        p = predicted[i, keep]
        kth = np.partition(t, k - 1)[k - 1]
        relevant = t <= kth + 1e-9 * max(1.0, abs(kth))
        order = np.lexsort((relevant.astype(np.int8), p))
        hits = relevant[order]
        precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
        aps[i] = precision[hits].sum() / hits.sum()
    return float(aps.mean())


def edge_map(edges: np.ndarray, predicted: np.ndarray) -> float:
    """Graph-reconstruction MAP (the paper's Table 1 MAP).

    For every node with at least one outgoing edge, rank all other nodes by
    predicted directed distance from it and take the average precision of its
    out-neighbours; exact ties are broken pessimistically.
    """

    n = edges.shape[0]
    aps = []
    for i in range(n):
        keep = np.arange(n) != i
        relevant = edges[i, keep]
        if not relevant.any():
            continue
        order = np.lexsort((relevant.astype(np.int8), predicted[i, keep]))
        hits = relevant[order]
        precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
        aps.append(float(precision[hits].mean()))
    return float(np.mean(aps)) if aps else 0.0


def hierarchy_alignment(predicted: np.ndarray, depths: np.ndarray) -> float:
    """Signed agreement between the predicted uphill potential and the hierarchy.

    Each node's potential is the total cost of reaching it minus the total cost
    of leaving it under the predicted distances. Over all pairs of nodes on
    different levels, +1 if the shallower node has the higher potential, -1 if
    lower, 0 if tied; symmetric distances give 0.
    """

    potential = predicted.sum(axis=0) - predicted.sum(axis=1)
    tol = 1e-9 * predicted.shape[0] * max(float(np.mean(np.abs(predicted))), 1e-12)
    dh = potential[:, None] - potential[None, :]
    dd = depths[None, :] - depths[:, None]
    cross = dd != 0
    sign_h = np.where(np.abs(dh) <= tol, 0.0, np.sign(dh))
    return float(np.sum(sign_h[cross] * np.sign(dd[cross])) / np.count_nonzero(cross))


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if not np.isfinite(value) else value


def tree_metrics(item: dict, predicted: np.ndarray) -> dict:
    target = item["distances"]
    off = ~np.eye(target.shape[0], dtype=bool)
    err = (predicted - target)[off]
    t = target[off]
    anti_target = (target - target.T)[off]
    anti_pred = (predicted - predicted.T)[off]
    sym_err = (0.5 * (predicted + predicted.T) - 0.5 * (target + target.T))[off]
    return {
        "normalized_stress": float(np.sqrt(np.sum(err**2) / np.sum(t**2))),
        "distortion": float(np.mean(err**2 / t)),
        "map_at_10": map_at_k(target, predicted),
        "asymmetry_rmse": _rms(anti_pred - anti_target),
        "asymmetry_correlation": _correlation(anti_target, anti_pred),
        "symmetric_rmse": _rms(sym_err),
        "hierarchy_alignment": hierarchy_alignment(predicted, item["depths"]),
    }


def graph_metrics(item: dict, predicted: np.ndarray) -> dict:
    target = item["distances"]
    off = ~np.eye(target.shape[0], dtype=bool)
    err = (predicted - target)[off]
    t = target[off]
    anti_target = (target - target.T)[off]
    anti_pred = (predicted - predicted.T)[off]
    sym_target = (0.5 * (target + target.T))[off]
    sym_pred = (0.5 * (predicted + predicted.T))[off]
    return {
        "normalized_stress": float(np.sqrt(np.sum(err**2) / np.sum(t**2))),
        "distortion": float(np.mean(err**2 / t)),
        "map": edge_map(item["edges"], predicted),
        "asymmetry_error": _rms(anti_pred - anti_target) / _rms(anti_target),
        "symmetric_error": _rms(sym_pred - sym_target) / _rms(sym_target),
    }


def instance_metrics(item: dict, predicted: np.ndarray) -> dict:
    return tree_metrics(item, predicted) if item["kind"] == "tree" else graph_metrics(item, predicted)


def floor_metrics(item: dict) -> dict:
    """Metrics of the empty (all-zero) prediction: the failure value."""

    target = item["distances"]
    off = ~np.eye(target.shape[0], dtype=bool)
    if item["kind"] == "tree":
        return {
            "normalized_stress": 1.0,
            "distortion": float(np.mean(target[off])),
            "map_at_10": 0.0,
            "asymmetry_rmse": _rms((target - target.T)[off]),
            "asymmetry_correlation": 0.0,
            "symmetric_rmse": _rms((0.5 * (target + target.T))[off]),
            "hierarchy_alignment": 0.0,
            "distance_preservation_pass": 0.0,
            "asymmetry_preservation_pass": 0.0,
            "hierarchy_direction_pass": 0.0,
        }
    return {
        "normalized_stress": 1.0,
        "distortion": float(np.mean(target[off])),
        "map": edge_map(item["edges"], np.zeros_like(target)),
        "asymmetry_error": 1.0,
        "symmetric_error": 1.0,
        "euclidean_gain_pass": 0.0,
    }


def claim_scores(item: dict, metrics: dict, baseline: dict | None) -> dict:
    scores = {}
    if item["kind"] == "graph":
        # Table 1 claim: lower distortion and higher MAP than the Euclidean
        # embedding of the same instance (the evaluator's frozen baseline).
        gain = (
            baseline is not None
            and metrics["distortion"] < baseline["distortion"]
            and metrics["map"] > baseline["map"]
        )
        scores["euclidean_gain_pass"] = float(gain)
        return scores
    for claim, (metric, threshold, direction) in CLAIM_THRESHOLDS.items():
        value = metrics[metric]
        if claim == "distance_preservation_pass":
            scores[claim] = float(np.clip(1.0 - value / threshold, 0.0, 1.0))
        elif direction == "le":
            scores[claim] = float(value <= threshold)
        else:
            scores[claim] = float(value >= threshold)
    return scores


def gate_failures(item: dict, metrics: dict) -> list[str]:
    failures = []
    gate = TREE_GATE if item["kind"] == "tree" else GRAPH_GATE
    for metric, (threshold, direction) in gate.items():
        value = metrics[metric]
        ok = value <= threshold if direction == "le" else value >= threshold
        if not ok:
            symbol = "<=" if direction == "le" else ">="
            failures.append(f"{metric}={value:.4g} fails {symbol} {threshold}")
    if item["kind"] == "graph":
        empty = item["floor"]
        for metric in ("normalized_stress", "distortion", "asymmetry_error"):
            if not metrics[metric] < empty[metric]:
                failures.append(f"{metric}={metrics[metric]:.4g} is not better than the empty prediction")
    return failures


# --------------------------------------------------------------------------
# Reading method outputs
# --------------------------------------------------------------------------


def read_prediction(path: Path, n_nodes: int, dim: int) -> np.ndarray:
    if not path.is_file():
        raise InvalidOutput("missing output file")
    try:
        with np.load(path, allow_pickle=False) as data:
            if "embedding" not in data.files:
                raise InvalidOutput("output has no 'embedding'")
            embedding = np.asarray(data["embedding"], dtype=np.float64)
            if "drift" in data.files:
                drift = np.asarray(data["drift"], dtype=np.float64).reshape(-1)
            elif "alpha" in data.files:
                alpha = np.asarray(data["alpha"], dtype=np.float64)
                if alpha.size != 1:
                    raise InvalidOutput("'alpha' must be a scalar")
                drift = np.zeros(dim)
                drift[-1] = float(alpha.reshape(()))
            else:
                raise InvalidOutput("output has neither 'drift' nor 'alpha'")
    except InvalidOutput:
        raise
    except Exception as exc:  # unreadable, pickled or non-numeric content
        raise InvalidOutput(f"unreadable output ({type(exc).__name__}: {exc})") from None
    if embedding.shape != (n_nodes, dim):
        raise InvalidOutput(f"embedding shape {embedding.shape} != ({n_nodes}, {dim})")
    if drift.shape != (dim,):
        raise InvalidOutput(f"drift shape {drift.shape} != ({dim},)")
    if not np.all(np.isfinite(embedding)) or not np.all(np.isfinite(drift)):
        raise InvalidOutput("non-finite embedding or drift")
    if float(np.linalg.norm(drift)) >= 1.0:
        raise InvalidOutput(f"drift norm {np.linalg.norm(drift):.4g} is not < 1")
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = randers_distances(embedding, drift)
    if not np.all(np.isfinite(predicted)):
        raise InvalidOutput("embedding produces non-finite distances")
    return predicted


# --------------------------------------------------------------------------
# Frozen Euclidean SMACOF baseline (always scored, never read from disk)
# --------------------------------------------------------------------------


def _classical_scaling(dissimilarities: np.ndarray, n_components: int) -> np.ndarray:
    d_sym = 0.5 * (dissimilarities + dissimilarities.T)
    n = d_sym.shape[0]
    centering = np.eye(n) - 1.0 / n
    gram = -0.5 * centering @ (d_sym**2) @ centering
    eigvals, eigvecs = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(eigvals)[::-1][:n_components]
    eigvecs = eigvecs[:, order]
    signs = np.sign(eigvecs[np.argmax(np.abs(eigvecs), axis=0), np.arange(n_components)])
    signs[signs == 0] = 1.0
    return eigvecs * signs * np.sqrt(np.clip(eigvals[order], 0.0, None))


def _euclidean_smacof_baseline(dissimilarities: np.ndarray, seed: int, dim: int) -> np.ndarray:
    """Uniform-weight Euclidean SMACOF. A symmetric distance can only fit the
    symmetric part of D, so it is fitted to (D + D^T) / 2 (the least-squares
    optimum of the directed stress over symmetric distances)."""

    d_sym = 0.5 * (dissimilarities + dissimilarities.T)
    n = d_sym.shape[0]
    rng = np.random.default_rng(seed)
    x = _classical_scaling(d_sym, dim)
    x = x + 0.05 * rng.standard_normal(x.shape)
    old_stress = None
    for _ in range(500):
        dist = np.linalg.norm(x[None, :, :] - x[:, None, :], axis=-1)
        stress = float(np.sum((dist - d_sym) ** 2))
        if old_stress is not None and abs(1.0 - stress / max(old_stress, 1e-12)) < 1e-5:
            break
        old_stress = stress
        dist[dist <= 1e-12] = 1e-12
        b = -d_sym / dist
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(axis=1))
        x = b @ x / n
        x -= x.mean(axis=0, keepdims=True)
    return randers_distances(x, np.zeros(dim))


# --------------------------------------------------------------------------
# Method discovery and execution
# --------------------------------------------------------------------------


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("._-") or "method"


def discover_cards() -> list[dict]:
    cards: list[dict] = []
    used = {BASELINE_NAME}
    if not CARD_ROOT.is_dir():
        return cards
    for path in sorted(CARD_ROOT.glob("**/method.json")):
        rel = str(path.relative_to(WORKSPACE))
        error = None
        command = None
        name = path.parent.name
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(spec, dict):
                raise ValueError("card must be a JSON object")
            command = spec.get("command")
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(part, str) and part for part in command)
            ):
                raise ValueError("'command' must be a non-empty list of strings")
            if isinstance(spec.get("name"), str) and spec["name"].strip():
                name = spec["name"]
        except Exception as exc:
            error = f"invalid method card: {exc}"
            command = None
        base = _safe_name(name)
        unique = base
        if unique in used:
            unique = f"{base}__{_safe_name(path.parent.name)}"
            index = 2
            while unique in used:
                unique = f"{base}__{_safe_name(path.parent.name)}_{index}"
                index += 1
        used.add(unique)
        cards.append({"name": unique, "card": rel, "command": command, "error": error})
    return cards


def write_instances(instances: dict[str, list[dict]], inst_dir: Path) -> None:
    inst_dir.mkdir(parents=True)
    entries = []
    for items in instances.values():
        for item in items:
            filename = f"{item['id']}.npz"
            np.savez_compressed(
                inst_dir / filename,
                dissimilarities=item["distances"],
                seed=np.array(item["method_seed"], dtype=np.int64),
            )
            entries.append(
                {"id": item["id"], "file": filename, "embedding_dim": EMBEDDING_DIMS[item["kind"]]}
            )
    (inst_dir / "index.json").write_text(json.dumps({"instances": entries}, indent=1))


def run_card(card: dict, inst_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True)
    command = list(card["command"]) + [
        "--instances",
        str(inst_dir),
        "--output",
        str(out_dir),
    ]
    record = {"returncode": None, "timed_out": False, "launch_error": None}
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=WORKSPACE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=METHOD_TIMEOUT_SECONDS,
        )
        record["returncode"] = proc.returncode
        log = proc.stdout or b""
    except subprocess.TimeoutExpired as exc:
        record["timed_out"] = True
        log = exc.stdout or b""
    except OSError as exc:
        record["launch_error"] = f"{type(exc).__name__}: {exc}"
        log = b""
    record["runtime_seconds_info_only"] = round(time.monotonic() - start, 2)
    record["log_tail"] = log[-4000:].decode("utf-8", errors="replace")
    return record


def _no_output(item: dict) -> np.ndarray:
    raise InvalidOutput("no output (method card unusable)")


def score_method(
    instances: dict[str, list[dict]],
    predict,
    run_failure: str | None,
    baseline_rows: dict | None,
) -> dict:
    """Score one method; ``predict(item)`` returns predicted distances or raises."""

    reasons: list[str] = []
    if run_failure:
        reasons.append(run_failure)
    raw: dict[str, list] = {}
    for experiment, items in instances.items():
        rows = []
        for item in items:
            try:
                predicted = predict(item)
            except InvalidOutput as exc:
                reasons.append(f"{item['id']}: {exc}")
                rows.append(None)
                continue
            metrics = instance_metrics(item, predicted)
            rows.append(metrics)
            reasons.extend(f"{item['id']}: {failure}" for failure in gate_failures(item, metrics))
        raw[experiment] = rows
    eligible = not reasons

    scores: dict[str, dict] = {}
    details: dict[str, dict] = {}
    for experiment, items in instances.items():
        rows = raw[experiment]
        if eligible:
            per_instance = [
                row | claim_scores(item, row, (baseline_rows or {}).get(item["id"]))
                for item, row in zip(items, rows)
            ]
        else:
            per_instance = [item["floor"] for item in items]
        scores[experiment] = {
            name: float(np.mean([row[name] for row in per_instance]))
            for name in per_instance[0]
        }
        names = TREE_RAW_METRICS if items[0]["kind"] == "tree" else GRAPH_RAW_METRICS
        complete = all(row is not None for row in rows)
        details[experiment] = {
            "raw_mean": (
                {name: float(np.mean([row[name] for row in rows])) for name in names}
                if complete
                else None
            ),
            "per_instance": {item["id"]: row for item, row in zip(items, rows)},
        }
    return {
        "eligible": eligible,
        "failure_reasons": reasons,
        "scores": scores,
        "details": details,
        "rows": {item["id"]: row for exp in instances for item, row in zip(instances[exp], raw[exp])},
    }


def _workers() -> int:
    try:
        return max(1, int(os.environ.get("GYM_EVAL_WORKERS", "1")))
    except ValueError:
        return 1


def evaluate_card(card: dict, instances: dict[str, list[dict]], baseline_rows: dict) -> dict:
    """Run one card on a private copy of the instances and score it."""

    print(f"[eval:{SLICE}] running {card['name']} ({card['card']})", flush=True)
    if card["error"]:
        result = score_method(instances, _no_output, card["error"], baseline_rows)
        result["run"] = {"card": card["card"]}
    else:
        with tempfile.TemporaryDirectory(prefix=f"gym_{SLICE}_") as tmp:
            inst_dir = Path(tmp) / "instances"
            out_dir = Path(tmp) / "embeddings"
            write_instances(instances, inst_dir)
            run = run_card(card, inst_dir, out_dir)
            failure = None
            if run["launch_error"]:
                failure = f"command could not start: {run['launch_error']}"
            elif run["timed_out"]:
                failure = f"command exceeded {METHOD_TIMEOUT_SECONDS} s"
            elif run["returncode"] != 0:
                failure = f"command exited with status {run['returncode']}"
            result = score_method(
                instances,
                lambda item: read_prediction(
                    out_dir / f"{item['id']}.npz",
                    item["distances"].shape[0],
                    EMBEDDING_DIMS[item["kind"]],
                ),
                failure,
                baseline_rows,
            )
        result["run"] = {"card": card["card"], "command": card["command"], **run}
    result.pop("rows", None)
    status = "eligible" if result["eligible"] else "FAILED"
    print(f"[eval:{SLICE}] {card['name']} -> {status}", flush=True)
    return result


def evaluate() -> tuple[dict, dict]:
    instances = build_instances()
    for items in instances.values():
        for item in items:
            item["floor"] = floor_metrics(item)
    results: dict[str, dict] = {}

    # The baseline is scored without a comparison row, so its own Euclidean
    # comparison claim is 0 (it cannot beat itself); its raw rows are the
    # comparison side for every other method.
    baseline = score_method(
        instances,
        lambda item: _euclidean_smacof_baseline(
            item["distances"], item["method_seed"], EMBEDDING_DIMS[item["kind"]]
        ),
        None,
        None,
    )
    baseline_rows = baseline.pop("rows")
    baseline["source"] = "frozen Euclidean SMACOF inside the evaluator"
    results[BASELINE_NAME] = baseline

    cards = discover_cards()
    # Methods run one after another unless GYM_EVAL_WORKERS > 1 (audit use).
    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        futures = [
            (card, pool.submit(evaluate_card, card, instances, baseline_rows)) for card in cards
        ]
        for card, future in futures:
            results[card["name"]] = future.result()

    scores = {
        experiment: {
            "results": {name: result["scores"][experiment] for name, result in results.items()}
        }
        for experiment in instances
    }
    details = {
        "slice": SLICE,
        "instances": "hidden",
        "instance_ids": {exp: [item["id"] for item in items] for exp, items in instances.items()},
        "tree_gate": {metric: list(rule) for metric, rule in TREE_GATE.items()},
        "graph_gate": {metric: list(rule) for metric, rule in GRAPH_GATE.items()},
        "embedding_dims": dict(EMBEDDING_DIMS),
        "claim_thresholds": {k: list(v) for k, v in CLAIM_THRESHOLDS.items()},
        "empty_prediction": {
            item["id"]: item["floor"] for items in instances.values() for item in items
        },
        "methods": {
            name: {
                "eligible": result["eligible"],
                "failure_reasons": result["failure_reasons"][:50],
                "n_failure_reasons": len(result["failure_reasons"]),
                "experiments": result["details"],
                **({"run": result["run"]} if "run" in result else {}),
                **({"source": result["source"]} if "source" in result else {}),
            }
            for name, result in results.items()
        },
    }
    return scores, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Hidden test-slice evaluator.")
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_test.json"))
    parser.add_argument("--details", type=Path, default=None)
    args = parser.parse_args()

    scores, details = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    details_path = args.details or args.output.with_name(f"details_{SLICE}.json")
    details_path.write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
