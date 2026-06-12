"""Visible train-slice scoring for directed-tree reconstruction artifacts.

This evaluator is method-agnostic: it reads prediction artifacts written by any
method that follows the workspace NPZ contract and scores only the visible
source-node half of each directed tree.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


SLICE = "train"
SPLIT_SEED = 250318010

DEPTH_TO_EXPERIMENT = {
    7: "directed_tree_reconstruction",
    5: "directed_tree_depth5_reconstruction",
}

CLAIM_THRESHOLDS = {
    "distance_preservation_pass": ("normalized_stress", 0.10, "le"),
    "asymmetry_preservation_pass": ("asymmetry_rmse", 0.10, "le"),
    "hierarchy_direction_pass": ("hierarchy_alignment", 0.80, "ge"),
}


def _artifact_text(artifact: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in artifact:
        return default
    value = artifact[key]
    if value.shape == ():
        return str(value.item())
    return str(value)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._-") or "method"


def _euclidean_distances(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x[None, :, :] - x[:, None, :], axis=-1)


def _randers_distances(x: np.ndarray, alpha: float) -> np.ndarray:
    delta = x[None, :, :] - x[:, None, :]
    return np.linalg.norm(delta, axis=-1) + alpha * delta[..., -1]


def _predicted_distances(
    artifact: np.lib.npyio.NpzFile, path: Path
) -> tuple[np.ndarray, np.ndarray | None]:
    if "predicted_distances" in artifact:
        predicted = artifact["predicted_distances"]
        embedding = artifact["embedding"] if "embedding" in artifact else None
        return predicted, embedding

    if "embedding" not in artifact:
        raise ValueError(f"{path} must contain either predicted_distances or embedding")

    embedding = artifact["embedding"]
    alpha = float(artifact["alpha"]) if "alpha" in artifact else 0.0
    distance_family = _artifact_text(artifact, "distance_family")
    method = _artifact_text(artifact, "method")
    if distance_family in {"randers", "finsler"} or method == "finsler_mds":
        predicted = _randers_distances(embedding, alpha)
    elif distance_family == "euclidean" or method == "euclidean_mds":
        predicted = _euclidean_distances(embedding)
    else:
        raise ValueError(
            f"{path} must specify distance_family='randers'/'euclidean', "
            "method='finsler_mds'/'euclidean_mds', or predicted_distances"
        )
    return predicted, embedding


def _source_indices(depths: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    for depth in np.unique(depths):
        nodes = np.flatnonzero(depths == depth)
        rng = np.random.default_rng(SPLIT_SEED + int(depth))
        permuted = np.array(nodes, copy=True)
        rng.shuffle(permuted)
        cutoff = (len(permuted) + 1) // 2
        selected.extend(int(node) for node in permuted[:cutoff])
    return np.array(sorted(selected), dtype=np.int64)


def _source_mask(target: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(target, dtype=bool)
    mask[source_indices, :] = True
    np.fill_diagonal(mask, False)
    return mask


def average_precision_from_distances(
    target: np.ndarray, predicted: np.ndarray, source_indices: np.ndarray, k: int = 10
) -> float:
    aps = []
    for i in source_indices:
        relevant = np.argsort(target[i])[1 : k + 1]
        relevant_set = set(int(j) for j in relevant)
        ranked = [int(j) for j in np.argsort(predicted[i]) if j != i]
        hits = 0
        precision_sum = 0.0
        for rank, node in enumerate(ranked, start=1):
            if node in relevant_set:
                hits += 1
                precision_sum += hits / rank
                if hits == k:
                    break
        aps.append(precision_sum / k)
    return float(np.mean(aps))


def metrics_for_artifact(path: Path) -> tuple[dict[str, float], dict[str, int | str]]:
    artifact = np.load(path, allow_pickle=False)
    target = artifact["distances"]
    depths = artifact["depths"] if "depths" in artifact else None
    if depths is None:
        raise ValueError(f"{path} must contain depths for source-node split scoring")

    predicted, embedding = _predicted_distances(artifact, path)
    if predicted.shape != target.shape:
        raise ValueError(
            f"{path} predicted_distances shape {predicted.shape} does not match "
            f"target distances shape {target.shape}"
        )

    sources = _source_indices(depths)
    mask = _source_mask(target, sources)
    raw_stress = float(np.sum((predicted[mask] - target[mask]) ** 2))
    denom = float(np.sum(target[mask] ** 2))
    normalized_stress = float(np.sqrt(raw_stress / denom))

    asym_error = float(
        np.sqrt(np.mean(((predicted - predicted.T) - (target - target.T))[mask] ** 2))
    )
    sym_error = float(
        np.sqrt(
            np.mean(
                (
                    (0.5 * (predicted + predicted.T))
                    - (0.5 * (target + target.T))
                )[mask]
                ** 2
            )
        )
    )
    if embedding is not None and len(sources) >= 3 and np.unique(depths[sources]).size > 1:
        rho = spearmanr(embedding[sources, -1], -depths[sources]).statistic
        hierarchy_alignment = float(0.0 if np.isnan(rho) else abs(rho))
    else:
        hierarchy_alignment = 0.0

    metrics = {
        "normalized_stress": normalized_stress,
        "map_at_10": average_precision_from_distances(
            target, predicted, sources, k=min(10, target.shape[0] - 1)
        ),
        "asymmetry_rmse": asym_error,
        "symmetric_rmse": sym_error,
        "hierarchy_alignment": hierarchy_alignment,
    }
    split_info = {
        "slice": SLICE,
        "split_seed": SPLIT_SEED,
        "source_node_count": int(len(sources)),
        "total_node_count": int(target.shape[0]),
    }
    return metrics, split_info


def claim_scores(metrics: dict[str, float]) -> dict[str, float]:
    scores = {}
    for claim, (metric_name, threshold, direction) in CLAIM_THRESHOLDS.items():
        value = metrics[metric_name]
        if np.isnan(value):
            scores[claim] = 0.0
            continue
        if direction == "le":
            passed = value <= threshold
        elif direction == "ge":
            passed = value >= threshold
        else:
            raise ValueError(f"unknown threshold direction: {direction}")
        scores[claim] = float(passed)
    return scores


def _experiment_for_artifact(path: Path) -> str | None:
    with np.load(path, allow_pickle=False) as artifact:
        if "depths" not in artifact:
            return None
        max_depth = int(np.max(artifact["depths"]))
    return DEPTH_TO_EXPERIMENT.get(max_depth)


def _method_name_for_artifact(path: Path, used_names: set[str]) -> str:
    with np.load(path, allow_pickle=False) as artifact:
        base = _artifact_text(artifact, "method", default=path.parent.name)
    name = _safe_name(base)
    if name not in used_names:
        return name
    parent_name = _safe_name(path.parent.name)
    candidate = f"{name}__{parent_name}"
    index = 2
    while candidate in used_names:
        candidate = f"{name}__{parent_name}_{index}"
        index += 1
    return candidate


def discover_manifest() -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for path in sorted(Path("outputs").glob("**/predictions.npz")):
        experiment = _experiment_for_artifact(path)
        if experiment is None:
            continue
        experiment_results = manifest.setdefault(experiment, {})
        method_name = _method_name_for_artifact(path, set(experiment_results))
        experiment_results[method_name] = str(path)
    if not manifest:
        raise FileNotFoundError(
            "no directed-tree prediction artifacts found under outputs/**/predictions.npz"
        )
    return manifest


def load_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return discover_manifest()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "experiments" in manifest:
        manifest = manifest["experiments"]
    normalized: dict[str, dict[str, str]] = {}
    for experiment, spec in manifest.items():
        if "results" in spec:
            spec = spec["results"]
        normalized[experiment] = {
            method_name: str(artifact_path)
            for method_name, artifact_path in spec.items()
        }
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional JSON mapping experiments to method artifact paths. "
            "Accepted shapes: {experiment: {method: path}} or "
            "{experiments: {experiment: {results: {method: path}}}}."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiment artifact groups to score.",
    )
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    experiments = args.experiments if args.experiments is not None else list(manifest)
    scores = {}
    details = {
        "slice": SLICE,
        "split_seed": SPLIT_SEED,
        "split_style": "stratified_source_node_half",
        "experiments": {},
    }
    for experiment in experiments:
        if experiment not in manifest:
            raise KeyError(f"{experiment} is not present in the evaluation manifest")
        result_paths = {name: Path(path) for name, path in manifest[experiment].items()}
        missing = [str(path) for path in result_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing artifacts for {experiment}: {', '.join(missing)}"
            )

        metric_results = {}
        split_results = {}
        for method_name, path in result_paths.items():
            metrics, split_info = metrics_for_artifact(path)
            metric_results[method_name] = metrics
            split_results[method_name] = split_info
        details["experiments"][experiment] = {
            "slice": SLICE,
            "results": metric_results,
            "split": split_results,
            "_thresholds": CLAIM_THRESHOLDS,
        }
        scores[experiment] = {
            "results": {
                method_name: claim_scores(metrics)
                for method_name, metrics in metric_results.items()
            }
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    details_path = out_path.with_name("details_train.json")
    with details_path.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
