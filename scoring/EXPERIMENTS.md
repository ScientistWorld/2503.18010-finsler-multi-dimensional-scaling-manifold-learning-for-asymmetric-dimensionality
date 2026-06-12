# Experiment Overview

## directed_tree_reconstruction
This experiment tests whether a two-dimensional representation can preserve directed graph distances on a hierarchical graph where moving down the hierarchy is cheaper than moving up. It measures overall distance reconstruction, directed-neighbour ranking quality, preservation of asymmetric distance components, and whether the recovered coordinate system reflects the hierarchy ordering.

## directed_tree_depth5_reconstruction
This experiment repeats the same directed-distance reconstruction test on a smaller hierarchy. It is intended to show whether the benchmark behavior persists when the graph size changes, using the same metrics and evaluation rules as the main directed-tree experiment.

The official `scores.json` values are binary checks of the qualitative Figure 5 claims. Continuous diagnostics are kept in `details.json` so future agents can inspect margins without treating non-paper numbers as paper references.
