# Overview

- **Paper ID:** 2503.18010
- **Title:** Finsler Multi-Dimensional Scaling: Manifold Learning for Asymmetric Dimensionality Reduction and Embedding
- **Domain:** Manifold learning / dimensionality reduction / directed graph embedding
- **TL;DR:** The paper extends MDS to asymmetric data by using a canonical Randers distance, enabling embeddings that preserve directed dissimilarities.

## Short Summary

Traditional MDS uses symmetric Euclidean or Riemannian distances, which cannot represent directed relationships. This paper defines Finsler MDS, using a canonical Randers space whose geodesics are still straight lines but whose distances depend on direction. It derives a SMACOF-style solver and also applies the same distance to neural directed-graph embeddings. Experiments show clearer asymmetric visualizations and better directed graph reconstruction/link prediction than Euclidean baselines.

## Key Results (paper)

- Table 1 (digraph reconstruction, neural embeddings of whole graphs): Finsler embeddings have lower distortion and higher MAP than Euclidean ones on every dataset and dimension, e.g. Cora in 10D: distortion `0.322 -> 0.093`, MAP `0.677 -> 0.931`; Citeseer in 10D: `0.305 -> 0.083`, `0.681 -> 0.958`.
- On the directed binary-tree visualization (Figure 5), the method reveals hierarchy direction while symmetric methods collapse or distort the structure.
- Link prediction AUC in Table 2 is competitive or best across the reported directed graph datasets.

## In This Gym

- Four experiments: the paper's directed trees (depth 7 and 5, 2D) and the paper's Cora and Citeseer citation digraphs (10D). The reference is the paper's Finsler SMACOF throughout.
- Hidden test slice, reference vs. the evaluator's Euclidean SMACOF: Cora distortion `0.135` vs `0.235`, MAP `0.440` vs `0.426`; Citeseer `0.129` vs `0.222`, MAP `0.492` vs `0.443`. The reference beats Euclidean on both on 7/10 Cora and 10/10 Citeseer subgraphs. On the trees it reaches stress about 0.02-0.03 with the hierarchy direction recovered; Euclidean cannot represent direction.
- Not reproduced: in 2D on the citation graphs, Finsler SMACOF loses to Euclidean on distortion and MAP (see `PROGRESS.md`); the gym therefore uses 10D, one of the paper's Table 1 dimensions.
