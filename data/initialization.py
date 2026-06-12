"""Shared initialization utilities for distance-embedding experiments."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import KernelPCA


def isomap_initialization(dissimilarities: np.ndarray, n_components: int) -> np.ndarray:
    """Classical scaling on the symmetrised shortest-path matrix.

    Appendix D.1.1 initializes SMACOF from an Isomap embedding of
    ``(D + D.T) / 2``. The benchmark already stores all-pairs shortest-path
    distances, so this helper performs the equivalent classical MDS step.
    """

    d_sym = 0.5 * (dissimilarities + dissimilarities.T)
    gram = -0.5 * d_sym**2
    return KernelPCA(n_components=n_components, kernel="precomputed").fit_transform(gram)
