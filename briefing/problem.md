# Problem Definition

## Research Question

Given a set of objects with pairwise dissimilarities that can be asymmetric, find a low-dimensional embedding whose directed pairwise distances preserve those dissimilarities. The benchmark focuses on directed graph distances, where the shortest path from node `i` to node `j` may differ from the reverse path.

## Why It Matters

Many relationships are directional: hierarchy, road or current travel time, citations, links, and information flow. A useful low-dimensional representation should preserve both proximity and directionality so that visualization, reconstruction, and link analysis do not discard the signal carried by asymmetry.

## Success Criteria

A successful system produces low-dimensional coordinates whose induced directed distances have low reconstruction error against the target directed shortest-path distances, preserve nearest-neighbour rankings, and retain the asymmetric component that a symmetric embedding cannot represent.
