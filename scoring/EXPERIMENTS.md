# Experiment Overview

## directed_tree_reconstruction
Tests whether a two-dimensional representation can preserve directed graph distances on a 255-node hierarchy in which moving down a level is cheap, moving up is expensive and nodes on a level are close to each other. It measures how well all directed distances are reproduced (globally and for short, local distances), whether the directional (asymmetric) part of the distances is kept, and whether the direction of the hierarchy can be read from the geometry. It is scored on the paper's exact tree (`scores.json`), on visible trees with per-level weights drawn around the paper's (`scores_train.json`), and on hidden trees from the same family (`scores_test.json`). A method must solve every instance to be ranked; otherwise it is scored as failed.

## directed_tree_depth5_reconstruction
The same measurement on 63-node (depth-5) hierarchies. It checks that the behaviour holds when the graph size changes, using the same metrics, instance families and eligibility rule.

The paper reports these experiments qualitatively. The reference file encodes the qualitative claims as pass checks, and the scored continuous quantities (stress and distortion) are the measures the paper uses for directed-distance reconstruction elsewhere.
