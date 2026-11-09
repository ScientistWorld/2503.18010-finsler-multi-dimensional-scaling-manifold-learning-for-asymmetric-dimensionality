# Research Direction

## Research Problem
Embed asymmetric dissimilarity data in two dimensions while preserving both proximity and direction. Improvements should reduce directed-distance reconstruction error (global stress and local distortion) while keeping the asymmetric component and the hierarchy direction intact on every instance.

## Core Contribution
The paper's contribution is to use a direction-dependent metric geometry for MDS-style embeddings. Future work should stay within asymmetric metric embeddings of this kind, i.e. two coordinates per node plus one global drift, and improve how such embeddings are found.

## Approach Scope
A scientist may improve the optimisation procedure (update rules, convergence, restarts), initialisation, weighting scheme, the choice of drift, or regularisation, as long as the method is a command that embeds unseen instances from their dissimilarity matrices alone.

## Out of Bounds
- Returning distances that are not induced by the submitted 2D coordinates and drift.
- Using more than two coordinates per node, per-node drift, or drift norm >= 1.
- Precomputing answers for specific instances, or using information other than the provided dissimilarities and seed.
- Changing the benchmark graphs, instance families, instance counts or evaluation code.
