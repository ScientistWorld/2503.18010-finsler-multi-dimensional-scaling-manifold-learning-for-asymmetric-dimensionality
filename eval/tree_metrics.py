"""Evaluation for directed-tree asymmetric distance reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


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


def metrics_for_artifact(path: Path) -> dict[str, float]:
    artifact = np.load(path, allow_pickle=False)
    embedding = artifact["embedding"]
    target = artifact["distances"]
    depths = artifact["depths"]
    alpha = float(artifact["alpha"])
    method = str(artifact["method"].item())

    if method == "finsler_mds":
        predicted = _randers_distances(embedding, alpha)
    elif method == "euclidean_mds":
        predicted = _euclidean_distances(embedding)
    else:
        raise ValueError(f"unknown method in artifact: {method}")

    weights = np.ones_like(target)
    np.fill_diagonal(weights, 0.0)
    raw_stress = float(np.sum(weights * (predicted - target) ** 2))
    denom = float(np.sum(weights * target**2))
    normalized_stress = float(np.sqrt(raw_stress / denom))
    asym_error = float(
        np.sqrt(np.mean(((predicted - predicted.T) - (target - target.T)) ** 2))
    )
    sym_error = float(
        np.sqrt(np.mean(((0.5 * (predicted + predicted.T)) - (0.5 * (target + target.T))) ** 2))
    )
    rho = spearmanr(embedding[:, -1], -depths).statistic
    hierarchy_alignment = float(0.0 if np.isnan(rho) else abs(rho))
    return {
        "normalized_stress": normalized_stress,
        "map_at_10": average_precision_from_distances(target, predicted, k=min(10, target.shape[0] - 1)),
        "asymmetry_rmse": asym_error,
        "symmetric_rmse": sym_error,
        "hierarchy_alignment": hierarchy_alignment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finsler", default="outputs/finsler_tree/predictions.npz")
    parser.add_argument("--euclidean", default="outputs/euclidean_tree/predictions.npz")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    scores = {
        "directed_tree_reconstruction": {
            "results": {
                "euclidean_mds": metrics_for_artifact(Path(args.euclidean)),
                "finsler_mds": metrics_for_artifact(Path(args.finsler)),
            }
        }
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
