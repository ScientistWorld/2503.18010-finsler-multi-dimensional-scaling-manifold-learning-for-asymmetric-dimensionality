"""Shared initialization utilities for distance-embedding experiments."""

from __future__ import annotations

import numpy as np


def isomap_initialization(dissimilarities: np.ndarray, n_components: int) -> np.ndarray:
    """Classical scaling on the symmetrised shortest-path matrix.

    Appendix D.1.1 initializes SMACOF from an Isomap embedding of
    ``(D + D.T) / 2``. The benchmark already provides all-pairs shortest-path
    distances, so this helper performs the equivalent classical MDS step
    (identical to scikit-learn's KernelPCA on ``-0.5 * D_sym**2``), using a
    dense eigendecomposition with a fixed sign convention so that the result is
    deterministic (the iterative ARPACK solver used before drew a random start
    vector and made reruns differ).
    """

    d_sym = 0.5 * (dissimilarities + dissimilarities.T)
    n = d_sym.shape[0]
    centering = np.eye(n) - 1.0 / n
    gram = -0.5 * centering @ (d_sym**2) @ centering
    gram = 0.5 * (gram + gram.T)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:n_components]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    pivots = np.argmax(np.abs(eigvecs), axis=0)
    signs = np.sign(eigvecs[pivots, np.arange(eigvecs.shape[1])])
    signs[signs == 0] = 1.0
    return eigvecs * signs * np.sqrt(eigvals)
