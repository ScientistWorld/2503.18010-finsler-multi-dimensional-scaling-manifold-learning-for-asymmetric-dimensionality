# Experiment Overview

## directed_tree_reconstruction
This experiment tests whether a two-dimensional representation can preserve directed graph distances on a hierarchical graph where moving down the hierarchy is cheaper than moving up. It measures overall distance reconstruction, directed-neighbour ranking quality, preservation of asymmetric distance components, and whether the recovered coordinate system reflects the hierarchy ordering.

## directed_tree_depth5_reconstruction
This experiment repeats the same directed-distance reconstruction test on a smaller hierarchy. It is intended to show whether the benchmark behavior persists when the graph size changes, using the same metrics and evaluation rules as the main directed-tree experiment.

The reference values for these experiments encode qualitative Figure 5 expectations as threshold checks because the paper does not publish numeric tree scores. The reproduced `scores.json` also includes continuous artifact-derived diagnostics so future agents can improve reconstruction quality instead of merely matching saturated pass/fail checks.
