# Finsler Multi-Dimensional Scaling: Manifold Learning for Asymmetric Dimensionality Reduction and Embedding

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2503.18010"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2503.18010-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2503.18010](https://arxiv.org/abs/2503.18010) |
|:---|:---|
| **Domain** | Manifold learning / dimensionality reduction / directed graph embedding |
| **TL;DR** | The paper extends MDS to asymmetric data by using a canonical Randers distance, enabling embeddings that preserve directed dissimilarities. |

## Summary

Traditional MDS uses symmetric Euclidean or Riemannian distances, which cannot represent directed relationships. This paper defines Finsler MDS, using a canonical Randers space whose geodesics are still straight lines but whose distances depend on direction. It derives a SMACOF-style solver and also applies the same distance to neural directed-graph embeddings. Experiments show clearer asymmetric visualizations and better directed graph reconstruction/link prediction than Euclidean baselines.

## Problem

Given a set of objects with pairwise dissimilarities that can be asymmetric, find a low-dimensional embedding whose directed pairwise distances preserve those dissimilarities. The benchmark focuses on directed graph distances, where the shortest path from node `i` to node `j` may differ from the reverse path.

## The paper's method

The paper generalizes multidimensional scaling to asymmetric dissimilarities by embedding data in a canonical Finsler, specifically Randers, space instead of a symmetric Riemannian/Euclidean space.

## Key results of the paper

- On Cora 2D directed-graph reconstruction, Table 1 reports distortion improving from `0.426` to `0.127` and MAP from `0.574` to `0.894` versus Euclidean embedding.
- On the directed binary-tree visualization, the method reveals hierarchy direction while symmetric methods collapse or distort the structure.
- Link prediction AUC in Table 2 is competitive or best across the reported directed graph datasets.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-terra-xhigh` | +122.7 |
| `gpt-5.5-xhigh` | +57.4 |
| `gpt-5.6-sol-xhigh` | +51.7 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2503.18010 --mode innovate --agent claude-code
sw evaluate 2503.18010
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

