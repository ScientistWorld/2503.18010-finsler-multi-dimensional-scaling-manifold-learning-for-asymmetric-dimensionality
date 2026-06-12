# Overview

- **Paper ID:** 2503.18010
- **Title:** Finsler Multi-Dimensional Scaling: Manifold Learning for Asymmetric Dimensionality Reduction and Embedding
- **Domain:** Manifold learning / dimensionality reduction / directed graph embedding
- **TL;DR:** The paper extends MDS to asymmetric data by using a canonical Randers distance, enabling embeddings that preserve directed dissimilarities.

## Short Summary

Traditional MDS uses symmetric Euclidean or Riemannian distances, which cannot represent directed relationships. This paper defines Finsler MDS, using a canonical Randers space whose geodesics are still straight lines but whose distances depend on direction. It derives a SMACOF-style solver and also applies the same distance to neural directed-graph embeddings. Experiments show clearer asymmetric visualizations and better directed graph reconstruction/link prediction than Euclidean baselines.

## Key Results

- On Cora 2D directed-graph reconstruction, Table 1 reports distortion improving from `0.426` to `0.127` and MAP from `0.574` to `0.894` versus Euclidean embedding.
- On the directed binary-tree visualization, the method reveals hierarchy direction while symmetric methods collapse or distort the structure.
- Link prediction AUC in Table 2 is competitive or best across the reported directed graph datasets.
