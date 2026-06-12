#!/usr/bin/env python
"""Run the Euclidean SMACOF baseline on the directed-tree benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baseline.euclidean_smacof import smacof
from data.directed_tree import make_directed_binary_tree
from method.finsler_smacof import isomap_initialization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/euclidean_tree")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = make_directed_binary_tree(depth=args.depth)
    rng = np.random.default_rng(args.seed)
    init = isomap_initialization(tree.distances, args.dim)
    init = init + 0.05 * rng.standard_normal(init.shape)
    result = smacof(
        tree.distances,
        n_components=args.dim,
        init=init,
        max_iter=args.max_iter,
        random_state=args.seed,
    )

    np.savez_compressed(
        out_dir / "predictions.npz",
        embedding=result.embedding,
        distances=tree.distances,
        adjacency=tree.adjacency,
        depths=tree.depths,
        alpha=np.array(0.0),
        method=np.array("euclidean_mds"),
        stress_history=np.array(result.stress_history, dtype=np.float64),
    )
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"n_iter": result.n_iter}, handle, indent=2)


if __name__ == "__main__":
    main()
