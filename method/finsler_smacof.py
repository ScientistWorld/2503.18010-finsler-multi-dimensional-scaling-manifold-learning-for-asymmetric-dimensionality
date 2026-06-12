"""Canonical Randers-space Finsler SMACOF.

This is a compact reproduction of Proposition 2 and Appendix D.1.1 from the
paper. The canonical drift is fixed to the final coordinate:
omega = (0, ..., 0, alpha)^T, alpha in [0, 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import gmres
from sklearn.decomposition import KernelPCA


@dataclass
class SmacofResult:
    embedding: np.ndarray
    stress_history: list[float]
    n_iter: int


def canonical_randers_distances(x: np.ndarray, alpha: float) -> np.ndarray:
    delta = x[None, :, :] - x[:, None, :]
    euclidean = np.linalg.norm(delta, axis=-1)
    return euclidean + alpha * delta[..., -1]


def euclidean_distances(x: np.ndarray) -> np.ndarray:
    delta = x[None, :, :] - x[:, None, :]
    return np.linalg.norm(delta, axis=-1)


def isomap_initialization(dissimilarities: np.ndarray, n_components: int) -> np.ndarray:
    """Classical scaling on the symmetrised shortest-path matrix."""

    d_sym = 0.5 * (dissimilarities + dissimilarities.T)
    gram = -0.5 * d_sym**2
    return KernelPCA(n_components=n_components, kernel="precomputed").fit_transform(gram)


def finsler_stress(
    x: np.ndarray,
    dissimilarities: np.ndarray,
    alpha: float,
    weights: np.ndarray | None = None,
) -> float:
    if weights is None:
        weights = np.ones_like(dissimilarities)
    diff = canonical_randers_distances(x, alpha) - dissimilarities
    return float(np.sum(weights * diff**2))


def smacof(
    dissimilarities: np.ndarray,
    alpha: float = 0.5,
    n_components: int = 2,
    init: np.ndarray | None = None,
    max_iter: int = 300,
    eps: float = 1e-5,
    weights: np.ndarray | None = None,
    random_state: int = 0,
    check_monotony: bool = False,
) -> SmacofResult:
    """Run Finsler SMACOF with the stable projected GMRES update."""

    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    if dissimilarities.shape[0] != dissimilarities.shape[1]:
        raise ValueError("dissimilarities must be square")

    n = dissimilarities.shape[0]
    if weights is None:
        weights = np.ones_like(dissimilarities)
    weights = weights.astype(np.float64, copy=True)
    np.fill_diagonal(weights, 0.0)

    if init is None:
        rng = np.random.default_rng(random_state)
        x = rng.random((n, n_components), dtype=np.float64)
    else:
        x = np.asarray(init, dtype=np.float64).copy()
        n_components = x.shape[1]

    v = -weights.copy()
    np.fill_diagonal(v, 0.0)
    np.fill_diagonal(v, weights.sum(axis=1))

    last_axis = np.zeros((n_components, n_components), dtype=np.float64)
    last_axis[-1, -1] = 1.0
    diag_sum = np.diag(weights.sum(axis=1))
    # With omega on the final axis, Eq. 29 reduces to V X + A X e_m e_m^T.
    a = alpha * (-weights + diag_sum)

    one_last_col = np.zeros((n, n_components), dtype=np.float64)
    one_last_col[:, -1] = 1.0
    c = alpha * ((weights * dissimilarities - weights.T * dissimilarities.T) @ one_last_col)

    lhs_left = v.T @ v
    lhs_asym = v.T @ a
    lhs = csr_array(np.kron(np.eye(n_components), lhs_left)) + csr_array(
        np.kron(last_axis.T, lhs_asym)
    )
    c_proj = v.T @ c

    history: list[float] = []
    old_x = x.copy()
    old_stress: float | None = None
    for iteration in range(max_iter):
        embedded = canonical_randers_distances(x, alpha)
        stress = float(np.sum(weights * (embedded - dissimilarities) ** 2))
        history.append(stress)

        if old_stress is not None:
            if check_monotony and stress > old_stress + 1e-6:
                x = old_x
                history[-1] = old_stress
                break
            if abs(1.0 - stress / max(old_stress, 1e-12)) < eps:
                break
        old_stress = stress
        old_x = x.copy()

        safe_embedded = embedded.copy()
        safe_embedded[safe_embedded <= 1e-12] = 1e-12
        b = -weights * dissimilarities / safe_embedded
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(axis=1))

        rhs = v.T @ (b @ x - c)
        solution, info = gmres(lhs, rhs.flatten(order="F"), atol=1e-8, rtol=1e-8)
        if info != 0:
            raise RuntimeError(f"GMRES failed to converge, info={info}")
        x = solution.reshape((n, n_components), order="F")
        x -= x.mean(axis=0, keepdims=True)

    return SmacofResult(embedding=x, stress_history=history, n_iter=len(history))
