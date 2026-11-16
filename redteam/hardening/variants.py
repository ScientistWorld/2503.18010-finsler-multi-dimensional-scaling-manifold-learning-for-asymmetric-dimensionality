#!/usr/bin/env python
"""Audit-only method variants, all behind the gym's method interface.

    python -m redteam.hardening.variants --variant <name> --instances DIR --output DIR

Variants (embedding dimension per instance from index.json: 2 for trees, 10
for citation graphs; canonical Randers drift along the last axis):

* ``exact_smacof``  - SMACOF that exactly majorizes the Finsler stress. The
  released update (method/finsler_smacof.py, copied from the authors' code)
  uses (1 + alpha) instead of (1 + alpha^2) on the drift axis and the Randers
  distance instead of the Euclidean norm inside B(X), so its fixed points are
  not stationary points of the stress. Here: X <- (B~(X) X - C/2) diag(1, 1 +
  alpha^2)^-1 / N with B~ built from the symmetrised dissimilarities and
  Euclidean norms (uniform weights). Same initialisation as the reference.
* ``lbfgs``         - the reference SMACOF result refined by L-BFGS on the
  same Finsler stress (analytic gradient), same alpha.
* ``multiscale``    - the reference SMACOF result refined by L-BFGS on the sum
  of the normalised stress (squared) and the 1/D-weighted stress, i.e. the
  global and the local (relative) distance criteria together.
* ``weighted``      - exact majorization (as ``exact_smacof``) of the stress
  with symmetric weights w_ij = 2 / (D_ij + D_ji), i.e. the paper's weighted
  Finsler stress with the 1/D weights of its Table 1 distortion.
* ``alpha_select``  - the reference SMACOF run for alpha in {0.3, ..., 0.7};
  keeps the embedding with the lowest normalised stress.
* ``random_init``   - the reference SMACOF from a uniform random start instead
  of the Isomap initialisation (weaker).
* ``isomap_only``   - closed form, no optimisation: the Isomap initialisation
  with drift (0, ..., 0, alpha) (weaker).
* ``level_collapse`` - degenerate probe: every node on one vertical line at the
  height given by the antisymmetric potential of the input; other axes 0.
* ``hybrid``        - closed-form probe: the first m - 1 classical-scaling
  coordinates of the symmetrised input, the antisymmetric potential as the
  last (drift) axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import pinv
from scipy.optimize import minimize

from data.initialization import isomap_initialization
from method.embed import embed_one
from method.finsler_smacof import smacof


def _init(dissimilarities: np.ndarray, seed: int, dim: int, noise: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    init = isomap_initialization(dissimilarities, dim)
    return init + noise * rng.standard_normal(init.shape)


def _stress(x: np.ndarray, d: np.ndarray, alpha: float) -> float:
    delta = x[None, :, :] - x[:, None, :]
    pred = np.linalg.norm(delta, axis=-1) + alpha * delta[..., -1]
    return float(np.sum((pred - d) ** 2))


def exact_smacof(
    d: np.ndarray, x0: np.ndarray, alpha: float, max_iter: int = 2000, eps: float = 1e-8
) -> tuple[np.ndarray, dict]:
    n, m = x0.shape
    sym = 0.5 * (d + d.T)
    c = (d - d.T).sum(axis=1)
    omega = np.zeros(m)
    omega[-1] = alpha
    axis_scale = 1.0 + omega**2
    x = x0 - x0.mean(axis=0, keepdims=True)
    history = [_stress(x, d, alpha)]
    for _ in range(max_iter):
        delta = x[None, :, :] - x[:, None, :]
        euclid = np.linalg.norm(delta, axis=-1)
        euclid[euclid <= 1e-12] = 1e-12
        b = -sym / euclid
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(axis=1))
        x = (b @ x - 0.5 * np.outer(c, omega)) / n / axis_scale
        x -= x.mean(axis=0, keepdims=True)
        history.append(_stress(x, d, alpha))
        if history[-1] > history[-2] * (1 + 1e-9) + 1e-12:
            raise RuntimeError("majorization increased the stress")
        if abs(1.0 - history[-1] / max(history[-2], 1e-12)) < eps:
            break
    return x, {"n_iter": len(history) - 1, "initial": history[0], "final": history[-1]}


def lbfgs_refine(d: np.ndarray, x0: np.ndarray, alpha: float, max_iter: int = 1500, multiscale: bool = False):
    n, m = x0.shape
    off = ~np.eye(n, dtype=bool)
    if multiscale:
        inv = np.zeros_like(d)
        inv[off] = 1.0 / d[off]
        pair_weight = 1.0 / np.sum(d[off] ** 2) + inv / np.sum(d[off])
    else:
        pair_weight = np.ones_like(d)
    np.fill_diagonal(pair_weight, 0.0)

    def fun(flat):
        x = flat.reshape(n, m)
        delta = x[None, :, :] - x[:, None, :]
        norm = np.linalg.norm(delta, axis=-1)
        err = norm + alpha * delta[..., -1] - d
        np.fill_diagonal(err, 0.0)
        unit = delta / np.maximum(norm[..., None], 1e-12)
        unit[..., -1] += alpha
        per_pair = 2.0 * (pair_weight * err)[..., None] * unit
        grad = per_pair.sum(axis=0) - per_pair.sum(axis=1)
        return float(np.sum(pair_weight * err**2)), grad.ravel()

    start = fun(x0.ravel())[0]
    res = minimize(
        fun,
        x0.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-9, "maxls": 50},
    )
    x = res.x.reshape(n, m)
    if res.fun > start:
        x = x0
    return x - x.mean(axis=0, keepdims=True), {"initial": start, "final": float(min(res.fun, start)), "n_iter": int(res.nit)}


def weighted_exact_smacof(
    d: np.ndarray, x0: np.ndarray, alpha: float, max_iter: int = 2000, eps: float = 1e-8
) -> tuple[np.ndarray, dict]:
    """Exact majorization of sum_ij w_ij (d_F(x_i, x_j) - D_ij)^2, w_ij = 1 / Dsym_ij.

    With V the weighted Laplacian, B(Z)_ij = -w_ij Dsym_ij / ||z_j - z_i|| and
    c_k = sum_i w_ik (D_ik - D_ki), the update is
    X <- V^+ (B(X) X + c omega^T / 2) (I + omega omega^T)^-1.
    """

    n, m = x0.shape
    sym = 0.5 * (d + d.T)
    off = ~np.eye(n, dtype=bool)
    w = np.zeros_like(d)
    w[off] = 1.0 / sym[off]
    lap = -w.copy()
    np.fill_diagonal(lap, w.sum(axis=1))
    lap_pinv = pinv(lap)
    c = (w * (d - d.T)).sum(axis=0)
    omega = np.zeros(m)
    omega[-1] = alpha
    axis_scale = 1.0 + omega**2

    def wstress(x):
        delta = x[None, :, :] - x[:, None, :]
        pred = np.linalg.norm(delta, axis=-1) + alpha * delta[..., -1]
        return float(np.sum(w * (pred - d) ** 2))

    x = x0 - x0.mean(axis=0, keepdims=True)
    history = [wstress(x)]
    for _ in range(max_iter):
        delta = x[None, :, :] - x[:, None, :]
        euclid = np.linalg.norm(delta, axis=-1)
        euclid[euclid <= 1e-12] = 1e-12
        b = -w * sym / euclid
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(axis=1))
        x = lap_pinv @ (b @ x + 0.5 * np.outer(c, omega)) / axis_scale
        x -= x.mean(axis=0, keepdims=True)
        history.append(wstress(x))
        if history[-1] > history[-2] * (1 + 1e-9) + 1e-12:
            raise RuntimeError("weighted majorization increased the stress")
        if abs(1.0 - history[-1] / max(history[-2], 1e-12)) < eps:
            break
    return x, {"n_iter": len(history) - 1, "initial": history[0], "final": history[-1]}


def alpha_select(d: np.ndarray, seed: int, dim: int) -> tuple[np.ndarray, float, dict]:
    off = ~np.eye(d.shape[0], dtype=bool)
    best = None
    for alpha in (0.3, 0.4, 0.5, 0.6, 0.7):
        x, _ = embed_one(d, seed, dim, alpha, 500, 1e-5, 0.05)
        delta = x[None, :, :] - x[:, None, :]
        pred = np.linalg.norm(delta, axis=-1) + alpha * delta[..., -1]
        stress = float(np.sqrt(np.sum((pred - d)[off] ** 2) / np.sum(d[off] ** 2)))
        if best is None or stress < best[2]:
            best = (x, alpha, stress)
    return best[0], best[1], {"alpha": best[1], "normalized_stress": best[2]}


def random_init(d: np.ndarray, seed: int, dim: int, alpha: float) -> np.ndarray:
    result = smacof(d, alpha=alpha, n_components=dim, init=None, max_iter=500, eps=1e-5,
                    random_state=seed, check_monotony=False)
    return result.embedding


def hybrid(d: np.ndarray, alpha: float, dim: int) -> np.ndarray:
    n = d.shape[0]
    x = isomap_initialization(d, dim - 1)
    potential = d.sum(axis=0) - d.sum(axis=1)
    return np.column_stack([x, potential / (2.0 * alpha * n)])


def level_collapse(d: np.ndarray, alpha: float, dim: int) -> np.ndarray:
    n = d.shape[0]
    potential = d.sum(axis=0) - d.sum(axis=1)
    y = potential / (2.0 * alpha * n)
    return np.column_stack([np.zeros((n, dim - 1)), y])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["exact_smacof", "lbfgs", "multiscale", "weighted", "alpha_select",
                                                  "random_init", "isomap_only", "level_collapse", "hybrid"])
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    index = json.loads((args.instances / "index.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in index["instances"]:
        dim = int(entry.get("embedding_dim", index.get("embedding_dim", 2)))
        with np.load(args.instances / entry["file"], allow_pickle=False) as data:
            d = np.asarray(data["dissimilarities"], dtype=np.float64)
            seed = int(data["seed"]) + args.seed_offset
        info: dict = {}
        alpha = args.alpha
        if args.variant == "exact_smacof":
            x, info = exact_smacof(d, _init(d, seed, dim), args.alpha)
        elif args.variant in ("lbfgs", "multiscale"):
            x_ref, _ = embed_one(d, seed, dim, args.alpha, 500, 1e-5, 0.05)
            x, info = lbfgs_refine(d, x_ref, args.alpha, multiscale=args.variant == "multiscale")
        elif args.variant == "weighted":
            x, info = weighted_exact_smacof(d, _init(d, seed, dim), args.alpha)
        elif args.variant == "alpha_select":
            x, alpha, info = alpha_select(d, seed, dim)
        elif args.variant == "random_init":
            x = random_init(d, seed, dim, args.alpha)
        elif args.variant == "isomap_only":
            x = isomap_initialization(d, dim)
        elif args.variant == "hybrid":
            x = hybrid(d, args.alpha, dim)
        else:
            x = level_collapse(d, args.alpha, dim)
        np.savez_compressed(
            args.output / f"{entry['id']}.npz",
            embedding=x,
            alpha=np.array(alpha),
            info=np.array(json.dumps(info)),
        )
        print(entry["id"], args.variant, info, flush=True)


if __name__ == "__main__":
    main()
