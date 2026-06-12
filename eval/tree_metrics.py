"""Evaluation for directed-tree asymmetric distance reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


DEFAULT_EXPERIMENTS = {
    "directed_tree_reconstruction": {
        "euclidean_mds": "outputs/euclidean_tree/predictions.npz",
        "finsler_mds": "outputs/finsler_tree/predictions.npz",
    },
    "directed_tree_depth5_reconstruction": {
        "euclidean_mds": "outputs/euclidean_tree_depth5/predictions.npz",
        "finsler_mds": "outputs/finsler_tree_depth5/predictions.npz",
    },
}

CLAIM_THRESHOLDS = {
    "distance_preservation_pass": ("normalized_stress", 0.10, "le"),
    "asymmetry_preservation_pass": ("asymmetry_rmse", 0.10, "le"),
    "hierarchy_direction_pass": ("hierarchy_alignment", 0.80, "ge"),
}


def _euclidean_distances(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x[None, :, :] - x[:, None, :], axis=-1)


def _randers_distances(x: np.ndarray, alpha: float) -> np.ndarray:
    delta = x[None, :, :] - x[:, None, :]
    return np.linalg.norm(delta, axis=-1) + alpha * delta[..., -1]


def average_precision_from_distances(target: np.ndarray, predicted: np.ndarray, k: int = 10) -> float:
    """MAP of nearest-neighbour ranking induced by predicted distances."""

    n = target.shape[0]
    aps = []
    for i in range(n):
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


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def _artifact_text(artifact: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in artifact:
        return default
    value = artifact[key]
    if value.shape == ():
        return str(value.item())
    return str(value)


def metrics_for_artifact(path: Path) -> dict[str, float]:
    artifact = np.load(path, allow_pickle=False)
    target = artifact["distances"]
    depths = artifact["depths"] if "depths" in artifact else None

    if "predicted_distances" in artifact:
        predicted = artifact["predicted_distances"]
        embedding = artifact["embedding"] if "embedding" in artifact else None
    else:
        if "embedding" not in artifact:
            raise ValueError(
                f"{path} must contain either predicted_distances or embedding"
            )
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

    if predicted.shape != target.shape:
        raise ValueError(
            f"{path} predicted_distances shape {predicted.shape} does not match "
            f"target distances shape {target.shape}"
        )

    weights = np.ones_like(target)
    np.fill_diagonal(weights, 0.0)
    raw_stress = float(np.sum(weights * (predicted - target) ** 2))
    denom = float(np.sum(weights * target**2))
    normalized_stress = float(np.sqrt(raw_stress / denom))
    asym_error = float(
        np.sqrt(np.mean(((predicted - predicted.T) - (target - target.T)) ** 2))
    )
    asymmetry_correlation = _safe_correlation(
        (target - target.T).ravel(),
        (predicted - predicted.T).ravel(),
    )
    sym_error = float(
        np.sqrt(np.mean(((0.5 * (predicted + predicted.T)) - (0.5 * (target + target.T))) ** 2))
    )
    if embedding is not None and depths is not None:
        rho = spearmanr(embedding[:, -1], -depths).statistic
        hierarchy_alignment = float(0.0 if np.isnan(rho) else abs(rho))
    else:
        hierarchy_alignment = 0.0
    return {
        "normalized_stress": normalized_stress,
        "map_at_10": average_precision_from_distances(target, predicted, k=min(10, target.shape[0] - 1)),
        "asymmetry_rmse": asym_error,
        "asymmetry_correlation": asymmetry_correlation,
        "symmetric_rmse": sym_error,
        "hierarchy_alignment": hierarchy_alignment,
    }


def claim_scores(metrics: dict[str, float]) -> dict[str, float]:
    """Convert continuous diagnostics into paper Figure 5 claim checks.

    Figure 5 reports the tree experiment qualitatively: Finsler MDS reveals
    hierarchy direction and preserves distances, while symmetric embeddings do
    not. The official scores therefore use binary checks instead of fabricated
    continuous paper targets. Full continuous diagnostics are written beside
    scores.json for analysis.
    """

    scores = {}
    for claim, (metric_name, threshold, direction) in CLAIM_THRESHOLDS.items():
        value = metrics[metric_name]
        if np.isnan(value):
            scores[claim] = 0.0
            continue
        if claim == "distance_preservation_pass" and direction == "le":
            scores[claim] = float(np.clip(1.0 - (value / threshold), 0.0, 1.0))
            continue
        if direction == "le":
            passed = value <= threshold
        elif direction == "ge":
            passed = value >= threshold
        else:
            raise ValueError(f"unknown threshold direction: {direction}")
        scores[claim] = float(passed)
    return scores


def official_scores(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "normalized_stress": metrics["normalized_stress"],
        "map_at_10": metrics["map_at_10"],
        "asymmetry_rmse": metrics["asymmetry_rmse"],
        "asymmetry_correlation": metrics["asymmetry_correlation"],
        "symmetric_rmse": metrics["symmetric_rmse"],
        "hierarchy_alignment": metrics["hierarchy_alignment"],
        **claim_scores(metrics),
    }


def load_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return DEFAULT_EXPERIMENTS
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
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    experiments = args.experiments if args.experiments is not None else list(manifest)
    scores = {}
    details = {}
    for experiment in experiments:
        if experiment not in manifest:
            raise KeyError(f"{experiment} is not present in the evaluation manifest")
        result_paths = {name: Path(path) for name, path in manifest[experiment].items()}
        missing = [str(path) for path in result_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing artifacts for {experiment}: {', '.join(missing)}"
            )

        metric_results = {
            method_name: metrics_for_artifact(path)
            for method_name, path in result_paths.items()
        }
        details[experiment] = {
            "results": metric_results,
            "_thresholds": CLAIM_THRESHOLDS,
        }
        scores[experiment] = {
            "results": {
                method_name: official_scores(metrics)
                for method_name, metrics in metric_results.items()
            }
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")
    details_path = out_path.with_name("details.json")
    with details_path.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
