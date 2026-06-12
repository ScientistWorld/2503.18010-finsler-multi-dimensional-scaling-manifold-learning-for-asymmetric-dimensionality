# Research Direction

## Research Problem
The benchmark asks how to embed asymmetric dissimilarity data in low dimensions while preserving both proximity and direction. Improvements should address directed distance reconstruction, not general graph drawing aesthetics alone.

## Core Contribution
The paper's core contribution is to use direction-dependent metric geometry for MDS-style embeddings. Future work should remain in the family of asymmetric metric embeddings or closely related directed-distance-preserving objectives.

## Approach Scope
A scientist may improve the optimization procedure, initialization, weighting scheme, distance parameterization, or regularization, as long as the submitted artifact induces directed pairwise distances from low-dimensional coordinates.

## Out of Bounds
- Replacing the benchmark with node classifiers, link classifiers, or graph neural networks that do not produce directed metric distances between all embedded node pairs.
- Using graph layout algorithms whose coordinates are not evaluated through a directed distance function.
- Increasing embedding dimensionality for the primary benchmark.
- Changing the benchmark graph or evaluation metrics to make the task easier.
