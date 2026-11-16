#!/usr/bin/env python
"""Audit-only shortcut submissions behind the gym's method interface.

    python -m redteam.hardening.shortcuts --kind <kind> --instances DIR --output DIR

Each kind replays a shortcut from REDTEAM.md (or a new one the hardened
interface must reject) against the hardened evaluators.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from method.embed import embed_one

KINDS = [
    "answer_copy",          # copy the input matrix as predicted distances; no fitted embedding
    "hierarchy_field",      # reference distances + an unrelated depth field as the embedding
    "task_failure",         # attenuate the asymmetric part until the asymmetry claim fails
    "score_fields",         # zero embedding plus self-reported perfect scores / flags
    "forged_targets",       # zero embedding plus forged target distances and depths
    "impossible_diagonal",  # reference embedding plus -inf self-distances
    "nan",                  # all-NaN embedding
    "inf",                  # all-infinite embedding
    "empty",                # empty embedding
    "wrong_dim",            # a three-dimensional Randers embedding (more capacity)
    "drift_too_large",      # reference embedding with drift norm >= 1 (not a metric)
    "missing_instance",     # reference embeddings, but one instance left out
    "crash",                # reference embeddings, then a non-zero exit status
    "generator_easy",       # coordinates built from the (patched) public generator
]


def _reference(d: np.ndarray, seed: int, dim: int = 2) -> np.ndarray:
    x, _ = embed_one(d, seed, dim, 0.5, 500, 1e-5, 0.05)
    return x


def _randers(x: np.ndarray, alpha: float) -> np.ndarray:
    delta = x[None, :, :] - x[:, None, :]
    return np.linalg.norm(delta, axis=-1) + alpha * delta[..., -1]


def _depth_field(d: np.ndarray) -> np.ndarray:
    """A hierarchy coordinate read off the input's antisymmetric potential."""
    n = d.shape[0]
    potential = d.sum(axis=0) - d.sum(axis=1)
    return np.column_stack([np.zeros(n), potential / n])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index = json.loads((args.instances / "index.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    entries = index["instances"]
    for position, entry in enumerate(entries):
        with np.load(args.instances / entry["file"], allow_pickle=False) as data:
            d = np.asarray(data["dissimilarities"], dtype=np.float64)
            seed = int(data["seed"])
        n = d.shape[0]
        out = args.output / f"{entry['id']}.npz"
        kind = args.kind
        if kind == "answer_copy":
            np.savez(out, predicted_distances=d.copy(), distances=d.copy(),
                     embedding=np.zeros((n, 2)), alpha=np.array(0.5))
        elif kind == "hierarchy_field":
            x = _reference(d, seed)
            np.savez(out, predicted_distances=_randers(x, 0.5),
                     embedding=_depth_field(d), alpha=np.array(0.5))
        elif kind == "task_failure":
            x = _reference(d, seed)
            np.savez(out, embedding=x, alpha=np.array(0.38))
        elif kind == "score_fields":
            fake = {m: np.array(1.0) for m in ["map_at_10", "hierarchy_alignment",
                                               "distance_preservation_pass",
                                               "asymmetry_preservation_pass",
                                               "hierarchy_direction_pass", "success", "eligible"]}
            fake.update({m: np.array(0.0) for m in ["normalized_stress", "distortion",
                                                    "asymmetry_rmse", "symmetric_rmse"]})
            np.savez(out, embedding=np.zeros((n, 2)), alpha=np.array(0.5), **fake)
        elif kind == "forged_targets":
            np.savez(out, embedding=np.zeros((n, 2)), alpha=np.array(0.5),
                     distances=np.zeros((n, n)), dissimilarities=np.zeros((n, n)),
                     depths=np.zeros(n))
        elif kind == "impossible_diagonal":
            x = _reference(d, seed)
            pred = _randers(x, 0.5)
            np.fill_diagonal(pred, -np.inf)
            np.savez(out, embedding=x, alpha=np.array(0.5), predicted_distances=pred)
        elif kind == "nan":
            np.savez(out, embedding=np.full((n, 2), np.nan), alpha=np.array(0.5))
        elif kind == "inf":
            np.savez(out, embedding=np.full((n, 2), np.inf), alpha=np.array(0.5))
        elif kind == "empty":
            np.savez(out, embedding=np.empty((0, 2)), alpha=np.array(0.5))
        elif kind == "wrong_dim":
            x = _reference(d, seed, dim=3)
            np.savez(out, embedding=x, alpha=np.array(0.5))
        elif kind == "drift_too_large":
            x = _reference(d, seed)
            np.savez(out, embedding=x / 2.0, alpha=np.array(1.0))
        elif kind == "missing_instance":
            if position == len(entries) - 1:
                continue
            np.savez(out, embedding=_reference(d, seed), alpha=np.array(0.5))
        elif kind == "crash":
            np.savez(out, embedding=_reference(d, seed), alpha=np.array(0.5))
        elif kind == "generator_easy":
            from data.directed_tree import make_directed_binary_tree

            depth = int(round(np.log2(n + 1))) - 1
            if 2 ** (depth + 1) - 1 != n:  # not a tree instance
                np.savez(out, embedding=_reference(d, seed), alpha=np.array(0.5))
                continue
            tree = make_directed_binary_tree(depth=depth)
            # Under a patched generator these are the "perfect" coordinates
            # for the generator's own node order.
            y = -np.asarray(tree.depths, dtype=float)
            np.savez(out, embedding=np.column_stack([np.arange(n) * 1e-4, y]),
                     alpha=np.array(0.5), distances=tree.distances)
    if args.kind == "crash":
        sys.exit(3)


if __name__ == "__main__":
    main()
