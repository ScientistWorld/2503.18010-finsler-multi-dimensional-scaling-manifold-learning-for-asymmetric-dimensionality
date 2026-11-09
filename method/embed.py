#!/usr/bin/env python
"""Finsler MDS (canonical Randers SMACOF) behind the gym's method interface.

The evaluators call

    python -m method.embed --instances <dir> --output <dir> [options]

where ``<dir>/index.json`` lists instance files holding a directed
``dissimilarities`` matrix and an integer ``seed``. For every instance this
writes ``<output>/<id>.npz`` with the two-dimensional ``embedding`` and the
scalar ``alpha`` of the canonical Randers drift (drift along the last axis),
following Appendix D.1.1: Isomap initialisation of the symmetrised
dissimilarities (plus small seeded noise), then Finsler SMACOF with
alpha = 0.5 and uniform weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.initialization import isomap_initialization
from method.finsler_smacof import smacof


def embed_one(
    dissimilarities: np.ndarray,
    seed: int,
    dim: int,
    alpha: float,
    max_iter: int,
    eps: float,
    init_noise: float,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    init = isomap_initialization(dissimilarities, dim)
    init = init + init_noise * rng.standard_normal(init.shape)
    result = smacof(
        dissimilarities,
        alpha=alpha,
        n_components=dim,
        init=init,
        max_iter=max_iter,
        eps=eps,
        random_state=seed,
        check_monotony=False,
    )
    return result.embedding, result.stress_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--init-noise", type=float, default=0.05)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Added to each instance seed (only used to measure seed spread).",
    )
    args = parser.parse_args()

    index = json.loads((args.instances / "index.json").read_text())
    dim = int(index.get("embedding_dim", 2))
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in index["instances"]:
        with np.load(args.instances / entry["file"], allow_pickle=False) as data:
            dissimilarities = np.asarray(data["dissimilarities"], dtype=np.float64)
            seed = int(data["seed"])
        embedding, history = embed_one(
            dissimilarities,
            seed + args.seed_offset,
            dim,
            args.alpha,
            args.max_iter,
            args.eps,
            args.init_noise,
        )
        np.savez_compressed(
            args.output / f"{entry['id']}.npz",
            embedding=embedding,
            alpha=np.array(args.alpha),
            stress_history=np.asarray(history, dtype=np.float64),
        )


if __name__ == "__main__":
    main()
