"""Visible (train) slice evaluator for the directed-hierarchy embedding gym.

Every method registered on disk is *run* on directed-tree instances and scored
from the two-dimensional embedding it returns. Nothing a method writes is
trusted except its coordinates and its drift vector; every distance and every
metric is recomputed here.

Method interface
----------------
A method registers itself with a card ``outputs/<any>/method.json``::

    {"name": "my_method", "command": ["python", "-m", "my_pkg.embed", "--flag"]}

The evaluator appends ``--instances <dir> --output <dir>`` and runs the command
from the workspace root. ``<dir>/index.json`` lists ``{"id", "file"}`` entries;
each instance file holds ``dissimilarities`` (N x N directed distances, row =
source) and ``seed`` (an int the method may use). For every instance the
method must write ``<output>/<id>.npz`` holding ``embedding`` (N x 2, finite)
and either ``drift`` (2-vector) or ``alpha`` (scalar, meaning drift (0, alpha)),
with drift norm < 1. Predicted directed distances are

    d(i -> j) = ||x_j - x_i||_2 + drift . (x_j - x_i)

so ``drift = 0`` is the ordinary Euclidean plane.

Slices
------
``--instances family`` (default): the visible per-level hierarchy family
(seeds listed below; regenerate them with ``data.directed_tree``).
``--instances paper``: the exact Appendix D.1.1 trees (depth 7 and depth 5).

Scoring
-------
A method *solves the task* only if every instance returns a valid embedding
and every instance clears the claim thresholds in ``GATE``. A method that does
not is scored as failed: every metric of every experiment is replaced by the
value of the empty (all-zero) prediction, which is worse than any eligible
value. The evaluator always adds its own frozen Euclidean SMACOF baseline under
the name ``euclidean_mds``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


SLICE = "train"
WORKSPACE = Path(__file__).resolve().parents[2]
CARD_ROOT = WORKSPACE / "outputs"
EMBEDDING_DIM = 2
METHOD_TIMEOUT_SECONDS = 3600
BASELINE_NAME = "euclidean_mds"
MAP_K = 10

EXPERIMENTS = {
    "directed_tree_reconstruction": {"depth": 7, "tag": "d7"},
    "directed_tree_depth5_reconstruction": {"depth": 5, "tag": "d5"},
}
# Visible instance seeds of the hierarchy family (data.directed_tree uses the
# same generator, so these trees can be rebuilt for development).
FAMILY_SEEDS = {
    "directed_tree_reconstruction": list(range(1000, 1008)),
    "directed_tree_depth5_reconstruction": list(range(2000, 2012)),
}
PAPER_METHOD_SEEDS = {
    "directed_tree_reconstruction": 0,
    "directed_tree_depth5_reconstruction": 1,
}

# Per-level hierarchy family around the paper's weights (0.5, 1.5, 0.1).
PAPER_DOWN, PAPER_UP, PAPER_SAME = 0.5, 1.5, 0.1
SCALE_RANGE = (0.8, 1.2)
UP_DOWN_RATIO_RANGE = (2.5, 3.0)
SAME_DEPTH_RANGE = (0.6, 1.4)

CLAIM_THRESHOLDS = {
    "distance_preservation_pass": ("normalized_stress", 0.10, "le"),
    "asymmetry_preservation_pass": ("asymmetry_rmse", 0.10, "le"),
    "hierarchy_direction_pass": ("hierarchy_alignment", 0.80, "ge"),
}
# Every instance must satisfy all of these for the method to be eligible.
GATE = {
    "normalized_stress": (0.10, "le"),
    "distortion": (0.10, "le"),
    "asymmetry_rmse": (0.10, "le"),
    "hierarchy_alignment": (0.80, "ge"),
}
RAW_METRICS = (
    "normalized_stress",
    "distortion",
    "map_at_10",
    "asymmetry_rmse",
    "asymmetry_correlation",
    "symmetric_rmse",
    "hierarchy_alignment",
)


class InvalidOutput(Exception):
    """A method output that cannot be scored."""


# --------------------------------------------------------------------------
# Instances
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


def build_instances(kind: str) -> dict[str, list[dict]]:
    instances: dict[str, list[dict]] = {}
    for exp_index, (experiment, spec) in enumerate(EXPERIMENTS.items()):
        depth = spec["depth"]
        items = []
        if kind == "paper":
            distances, levels = _tree_distances(
                depth,
                np.full(depth, PAPER_DOWN),
                np.full(depth, PAPER_UP),
                np.full(depth, PAPER_SAME),
            )
            items.append(
                {
                    "id": f"{spec['tag']}_paper",
                    "distances": distances,
                    "depths": levels,
                    "method_seed": PAPER_METHOD_SEEDS[experiment],
                }
            )
        elif kind == "family":
            for k, seed in enumerate(FAMILY_SEEDS[experiment]):
                distances, levels = _family_tree(depth, seed)
                items.append(
                    {
                        "id": f"{spec['tag']}_{k:02d}",
                        "distances": distances,
                        "depths": levels,
                        "method_seed": 100 * exp_index + k,
                    }
                )
        else:
            raise ValueError(f"unknown instance kind {kind!r}")
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


def instance_metrics(target: np.ndarray, depths: np.ndarray, predicted: np.ndarray) -> dict:
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
        "hierarchy_alignment": hierarchy_alignment(predicted, depths),
    }


def floor_metrics(target: np.ndarray) -> dict:
    """Metrics of the empty (all-zero) prediction: the failure value."""

    off = ~np.eye(target.shape[0], dtype=bool)
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


def claim_scores(metrics: dict) -> dict:
    scores = {}
    for claim, (metric, threshold, direction) in CLAIM_THRESHOLDS.items():
        value = metrics[metric]
        if claim == "distance_preservation_pass":
            scores[claim] = float(np.clip(1.0 - value / threshold, 0.0, 1.0))
        elif direction == "le":
            scores[claim] = float(value <= threshold)
        else:
            scores[claim] = float(value >= threshold)
    return scores


def gate_failures(metrics: dict) -> list[str]:
    failures = []
    for metric, (threshold, direction) in GATE.items():
        value = metrics[metric]
        ok = value <= threshold if direction == "le" else value >= threshold
        if not ok:
            symbol = "<=" if direction == "le" else ">="
            failures.append(f"{metric}={value:.4g} fails {symbol} {threshold}")
    return failures


# --------------------------------------------------------------------------
# Reading method outputs
# --------------------------------------------------------------------------


def read_prediction(path: Path, n_nodes: int) -> np.ndarray:
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
                drift = np.array([0.0, float(alpha.reshape(()))])
            else:
                raise InvalidOutput("output has neither 'drift' nor 'alpha'")
    except InvalidOutput:
        raise
    except Exception as exc:  # unreadable, pickled or non-numeric content
        raise InvalidOutput(f"unreadable output ({type(exc).__name__}: {exc})") from None
    if embedding.shape != (n_nodes, EMBEDDING_DIM):
        raise InvalidOutput(
            f"embedding shape {embedding.shape} != ({n_nodes}, {EMBEDDING_DIM})"
        )
    if drift.shape != (EMBEDDING_DIM,):
        raise InvalidOutput(f"drift shape {drift.shape} != ({EMBEDDING_DIM},)")
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


def _euclidean_smacof_baseline(dissimilarities: np.ndarray, seed: int) -> np.ndarray:
    n = dissimilarities.shape[0]
    rng = np.random.default_rng(seed)
    x = _classical_scaling(dissimilarities, EMBEDDING_DIM)
    x = x + 0.05 * rng.standard_normal(x.shape)
    old_stress = None
    for _ in range(500):
        dist = np.linalg.norm(x[None, :, :] - x[:, None, :], axis=-1)
        stress = float(np.sum((dist - dissimilarities) ** 2))
        if old_stress is not None and abs(1.0 - stress / max(old_stress, 1e-12)) < 1e-5:
            break
        old_stress = stress
        dist[dist <= 1e-12] = 1e-12
        b = -dissimilarities / dist
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(axis=1))
        x = b @ x / n
        x -= x.mean(axis=0, keepdims=True)
    return randers_distances(x, np.zeros(EMBEDDING_DIM))


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


def run_card(card: dict, instances: dict[str, list[dict]], scratch: Path) -> dict:
    inst_dir = scratch / "instances"
    out_dir = scratch / "embeddings"
    inst_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    entries = []
    for items in instances.values():
        for item in items:
            filename = f"{item['id']}.npz"
            np.savez(
                inst_dir / filename,
                dissimilarities=item["distances"],
                seed=np.array(item["method_seed"], dtype=np.int64),
            )
            entries.append({"id": item["id"], "file": filename})
    (inst_dir / "index.json").write_text(
        json.dumps({"embedding_dim": EMBEDDING_DIM, "instances": entries}, indent=1)
    )
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
    record["output_dir"] = out_dir
    return record


def _no_output(item: dict) -> np.ndarray:
    raise InvalidOutput("no output (method card unusable)")


def score_method(instances: dict[str, list[dict]], predict, run_failure: str | None) -> dict:
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
            metrics = instance_metrics(item["distances"], item["depths"], predicted)
            rows.append(metrics)
            reasons.extend(f"{item['id']}: {failure}" for failure in gate_failures(metrics))
        raw[experiment] = rows
    eligible = not reasons

    scores: dict[str, dict] = {}
    details: dict[str, dict] = {}
    for experiment, items in instances.items():
        rows = raw[experiment]
        if eligible:
            per_instance = [row | claim_scores(row) for row in rows]
        else:
            per_instance = [floor_metrics(item["distances"]) for item in items]
        scores[experiment] = {
            name: float(np.mean([row[name] for row in per_instance]))
            for name in per_instance[0]
        }
        complete = all(row is not None for row in rows)
        details[experiment] = {
            "raw_mean": (
                {name: float(np.mean([row[name] for row in rows])) for name in RAW_METRICS}
                if complete
                else None
            ),
            "per_instance": {item["id"]: row for item, row in zip(items, rows)},
        }
    return {"eligible": eligible, "failure_reasons": reasons, "scores": scores, "details": details}


def evaluate(kind: str) -> tuple[dict, dict]:
    instances = build_instances(kind)
    results: dict[str, dict] = {}

    baseline = score_method(
        instances,
        lambda item: _euclidean_smacof_baseline(item["distances"], item["method_seed"]),
        None,
    )
    baseline["source"] = "frozen Euclidean SMACOF inside the evaluator"
    results[BASELINE_NAME] = baseline

    for card in discover_cards():
        print(f"[eval:{SLICE}] running {card['name']} ({card['card']})", flush=True)
        if card["error"]:
            result = score_method(instances, _no_output, card["error"])
            result["run"] = {"card": card["card"]}
        else:
            with tempfile.TemporaryDirectory(prefix=f"gym_{SLICE}_") as tmp:
                run = run_card(card, instances, Path(tmp))
                failure = None
                if run["launch_error"]:
                    failure = f"command could not start: {run['launch_error']}"
                elif run["timed_out"]:
                    failure = f"command exceeded {METHOD_TIMEOUT_SECONDS} s"
                elif run["returncode"] != 0:
                    failure = f"command exited with status {run['returncode']}"
                out_dir = run.pop("output_dir")
                result = score_method(
                    instances,
                    lambda item: read_prediction(
                        out_dir / f"{item['id']}.npz", item["distances"].shape[0]
                    ),
                    failure,
                )
            result["run"] = {"card": card["card"], "command": card["command"], **run}
        results[card["name"]] = result
        status = "eligible" if result["eligible"] else "FAILED"
        print(f"[eval:{SLICE}]   -> {status}", flush=True)

    scores = {
        experiment: {
            "results": {name: result["scores"][experiment] for name, result in results.items()}
        }
        for experiment in instances
    }
    details = {
        "slice": SLICE,
        "instances": kind,
        "instance_ids": {exp: [item["id"] for item in items] for exp, items in instances.items()},
        "gate": {metric: list(rule) for metric, rule in GATE.items()},
        "claim_thresholds": {k: list(v) for k, v in CLAIM_THRESHOLDS.items()},
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instances", choices=["family", "paper"], default="family")
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_train.json"))
    parser.add_argument("--details", type=Path, default=None)
    args = parser.parse_args()

    scores, details = evaluate(args.instances)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    details_path = args.details
    if details_path is None:
        stem = "details" if args.instances == "paper" else f"details_{SLICE}"
        details_path = args.output.with_name(f"{stem}.json")
    details_path.write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
