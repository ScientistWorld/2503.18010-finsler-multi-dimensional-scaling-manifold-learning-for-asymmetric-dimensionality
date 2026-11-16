# Research Direction

## Research Problem
Embed asymmetric dissimilarity data in a low-dimensional space while preserving both proximity and direction: synthetic directed hierarchies in 2D and real citation digraphs (Cora, Citeseer) in 10D. Improvements should reduce directed-distance reconstruction error (global stress and local distortion) and, on the citation graphs, raise graph-reconstruction MAP, while keeping the asymmetric component (and, on the trees, the hierarchy direction) intact on every instance.

## Core Contribution
The paper's contribution is to use a direction-dependent metric geometry for MDS-style embeddings. Future work should stay within asymmetric metric embeddings of this kind, i.e. the given number of coordinates per node plus one global drift, and improve how such embeddings are found.

## Approach Scope
A scientist may improve the optimisation procedure (update rules, convergence, restarts), initialisation, weighting scheme, the choice of drift, or regularisation, as long as the method is a command that embeds unseen instances from their dissimilarity matrices alone. The hardening audit (`HARDENING.md`) shows clear headroom above the reference on the citation experiments.

## Out of Bounds
- Returning distances that are not induced by the submitted coordinates and drift.
- Using a different number of coordinates per node than `embedding_dim`, per-node drift, or drift norm >= 1.
- Precomputing answers for specific instances, or using information other than the provided dissimilarities, embedding dimension and seed.
- Changing the benchmark graphs, instance families, instance counts or evaluation code.
