"""Euclidean SMACOF baseline for the directed-tree benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import pinv


@dataclass
class EuclideanResult:
    embedding: np.ndarray
    stress_history: list[float]
    n_iter: int


def euclidean_distances(x: np.ndarray) -> np.ndarray:
    delta = x[None, :, :] - x[:, None, :]
    return np.linalg.norm(delta, axis=-1)


def smacof(
    dissimilarities: np.ndarray,
    n_components: int = 2,
    init: np.ndarray | None = None,
    max_iter: int = 300,
    eps: float = 1e-5,
    weights: np.ndarray | None = None,
    random_state: int = 0,
) -> EuclideanResult:
    if weights is None:
        weights = np.ones_like(dissimilarities)
    weights = weights.astype(np.float64, copy=True)
    np.fill_diagonal(weights, 0.0)
    n = dissimilarities.shape[0]

    if init is None:
        rng = np.random.default_rng(random_state)
        x = rng.random((n, n_components), dtype=np.float64)
    else:
        x = np.asarray(init, dtype=np.float64).copy()

    v = -weights.copy()
    np.fill_diagonal(v, 0.0)
    np.fill_diagonal(v, weights.sum(axis=1))
    v_pinv = pinv(v)

    history: list[float] = []
    old_stress: float | None = None
    for iteration in range(max_iter):
        embedded = euclidean_distances(x)
        stress = float(np.sum(weights * (embedded - dissimilarities) ** 2))
        history.append(stress)
        if old_stress is not None and abs(1.0 - stress / max(old_stress, 1e-12)) < eps:
            break
        old_stress = stress

        safe_embedded = embedded.copy()
        safe_embedded[safe_embedded <= 1e-12] = 1e-12
        b = -weights * dissimilarities / safe_embedded
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(axis=1))
        x = v_pinv @ b @ x
        x -= x.mean(axis=0, keepdims=True)

    return EuclideanResult(embedding=x, stress_history=history, n_iter=len(history))
