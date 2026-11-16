#!/usr/bin/env python
"""Euclidean SMACOF baseline behind the gym's method interface.

    python -m baseline.embed --instances <dir> --output <dir>

Writes ``<output>/<id>.npz`` with an ``embedding`` of the instance's
``embedding_dim`` and a zero drift (``alpha = 0``), i.e. ordinary symmetric Euclidean distances. The
evaluators always score their own frozen copy of this baseline under the name
``euclidean_mds``; this public copy is a template for the method interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baseline.euclidean_smacof import smacof
from data.initialization import isomap_initialization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--init-noise", type=float, default=0.05)
    args = parser.parse_args()

    index = json.loads((args.instances / "index.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in index["instances"]:
        dim = int(entry.get("embedding_dim", index.get("embedding_dim", 2)))
        with np.load(args.instances / entry["file"], allow_pickle=False) as data:
            dissimilarities = np.asarray(data["dissimilarities"], dtype=np.float64)
            seed = int(data["seed"])
        rng = np.random.default_rng(seed)
        init = isomap_initialization(dissimilarities, dim)
        init = init + args.init_noise * rng.standard_normal(init.shape)
        result = smacof(
            dissimilarities,
            n_components=dim,
            init=init,
            max_iter=args.max_iter,
            eps=args.eps,
            random_state=seed,
        )
        np.savez_compressed(
            args.output / f"{entry['id']}.npz",
            embedding=result.embedding,
            alpha=np.array(0.0),
        )


if __name__ == "__main__":
    main()
