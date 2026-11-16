#!/usr/bin/env python
"""Build the processed citation-digraph edge lists from the raw LINQS files.

Pure Python (no third-party packages), so it runs anywhere:

    python data/build_citation_graphs.py            # write data/citation_graphs/*.txt
    python data/build_citation_graphs.py --check    # verify the tracked copies

Input (fetched by ``scripts/download.sh``):

* ``data/downloads/cora/cora.cites``          (LINQS Cora, 2,708 papers, 5,429 citations)
* ``data/downloads/citeseer/citeseer.cites``  (LINQS CiteSeer, 4,732 citation lines)

Each line is ``<cited paper> <citing paper>``; the directed edge runs from the
citing paper to the cited paper. Self-citations and duplicate lines are
dropped. Only the largest weakly connected component is kept, its nodes are
numbered 0..n-1 in the sorted order of their original paper IDs, and the edge
list is written sorted. The output format is::

    # comment lines
    n_nodes <n>
    <src> <dst>        # one directed edge (citing -> cited) per line

The evaluators ship their own copies of these files (``eval/train/graphs/``
and ``eval/test/graphs/``) and check their SHA-256; ``--check`` confirms the
public copy, the evaluator copies and a fresh build from the raw files agree.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = {
    "cora": ROOT / "data" / "downloads" / "cora" / "cora.cites",
    "citeseer": ROOT / "data" / "downloads" / "citeseer" / "citeseer.cites",
}
RAW_SHA256 = {
    "cora": "ec1a372391b7f0f60a6aff0084e8abd8f19f0faa7e1f2441a41c492042d5945e",
    "citeseer": "cd9730630eaaa745058c2cd2cce6b9afb79ca71278a479aebec55f4cf5eb7f30",
}
OUT_DIR = ROOT / "data" / "citation_graphs"
EVAL_COPIES = [ROOT / "eval" / "train" / "graphs", ROOT / "eval" / "test" / "graphs"]


def build(name: str) -> str:
    raw_path = RAW[name]
    raw = raw_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RAW_SHA256[name]:
        raise SystemExit(f"{raw_path}: sha256 {digest} does not match the expected LINQS file")
    edges = set()
    for line in raw.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        cited, citing = parts
        if cited != citing:
            edges.add((citing, cited))
    nodes = sorted({v for e in edges for v in e})
    neighbours = {v: [] for v in nodes}
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    seen: set[str] = set()
    best: list[str] = []
    for start in nodes:
        if start in seen:
            continue
        component = [start]
        seen.add(start)
        head = 0
        while head < len(component):
            u = component[head]
            head += 1
            for v in neighbours[u]:
                if v not in seen:
                    seen.add(v)
                    component.append(v)
        if len(component) > len(best):
            best = component
    keep = sorted(best)
    index = {v: i for i, v in enumerate(keep)}
    kept = sorted((index[a], index[b]) for a, b in edges if a in index and b in index)
    lines = [
        f"# {name}: largest weakly connected component of the LINQS citation graph",
        f"# source sha256 {RAW_SHA256[name]}",
        "# nodes numbered in sorted order of the original paper IDs; edge = citing -> cited",
        f"n_nodes {len(keep)}",
    ]
    lines += [f"{a} {b}" for a, b in kept]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ok = True
    for name in RAW:
        text = build(name)
        digest = hashlib.sha256(text.encode()).hexdigest()
        target = OUT_DIR / f"{name}_lwcc_edges.txt"
        if args.check:
            for folder in [OUT_DIR] + EVAL_COPIES:
                path = folder / target.name
                same = path.is_file() and path.read_text() == text
                ok &= same
                print(f"{path.relative_to(ROOT)}: {'ok' if same else 'MISMATCH'}")
        else:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            n_edges = text.count("\n") - 4
            print(f"{target.relative_to(ROOT)}: sha256 {digest} ({n_edges} edges)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
