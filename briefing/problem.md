# Problem Definition

## Research Question

Given a set of objects with pairwise dissimilarities that can be asymmetric, find a low-dimensional embedding whose directed pairwise distances preserve those dissimilarities. The benchmark uses directed graph distances, where the shortest path from node `i` to node `j` may differ from the reverse path.

## The Benchmark Graphs

Each instance is a complete binary tree (depth 7: 255 nodes, or depth 5: 63 nodes) whose directed edges model a hierarchy: moving down a level (parent to child) is cheap, moving up (child to parent) costs about 2.5-3 times more, and all nodes on the same level are linked by short undirected edges. Directed shortest-path distances on this graph form the dissimilarity matrix `D` (row = source node).

- The paper's exact setting uses weights 0.5 (down), 1.5 (up) and 0.1 (same level) on every level.
- The train and test instances keep the same topology but draw each level's weights around those values (down-edge scale in [0.8, 1.2] x 0.5, up/down ratio in [2.5, 3.0], same-level weight in [0.6, 1.4] x 0.1) and shuffle the node labels, so node indices carry no information. `data/directed_tree.py` (`sample_hierarchy_instance`) generates trees from this family for development.

A method receives only `D` (and a seed). It does not receive the tree, the levels or the edge weights.

## What a Solution Returns

Two-dimensional coordinates `x_i` for every node and one constant drift vector `w` with `||w|| < 1`. Directed distances are then read off as

    d(i -> j) = ||x_j - x_i|| + w . (x_j - x_i)

which is the ordinary Euclidean plane when `w = 0`. The evaluator computes every distance from these coordinates itself.

## Why It Matters

Many relationships are directional: hierarchy, travel time against a current, citations, links, information flow. A useful low-dimensional representation should preserve both proximity and direction, so that visualisation and reconstruction do not discard the signal carried by asymmetry.

## Success Criteria

A successful method produces, on every instance, coordinates whose induced directed distances
- reproduce the target distances globally and locally (low stress and low distortion),
- retain the asymmetric component that a symmetric embedding cannot represent, and
- place nodes so that the direction of the hierarchy can be read from the geometry.
