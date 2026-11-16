# Experiment Overview

## directed_tree_reconstruction
Tests whether a two-dimensional representation can preserve directed graph distances on a 255-node hierarchy in which moving down a level is cheap, moving up is expensive and nodes on a level are close to each other. It measures how well all directed distances are reproduced (globally and for short, local distances), whether the directional (asymmetric) part of the distances is kept, and whether the direction of the hierarchy can be read from the geometry. It is scored on the paper's exact tree (`scores.json`), on visible trees with per-level weights drawn around the paper's (`scores_train.json`), and on hidden trees from the same family (`scores_test.json`). A method must solve every instance of every experiment to be ranked; otherwise it is scored as failed.

## directed_tree_depth5_reconstruction
The same measurement on 63-node (depth-5) hierarchies. It checks that the behaviour holds when the graph size changes, using the same metrics, instance families and eligibility rule.

## cora_digraph_reconstruction
Tests the paper's central quantitative claim (Table 1) on a real directed graph: that an asymmetric representation reconstructs the directed distances of the Cora citation network with lower distortion and higher graph-reconstruction precision (MAP over cited papers) than a symmetric Euclidean representation of the same size. Directed distances follow citations cheaply and travel against them at three times the cost, so every pair has a finite, asymmetric distance. Each node gets a 10-dimensional representation (one of the paper's Table 1 dimensions). It is scored on the whole largest connected component (`scores.json`, 2,485 papers) and on 400-paper connected subgraphs with shuffled labels (5 visible, 10 hidden). Besides distortion and MAP it measures global stress, how much of the directional part of the distances is kept (a symmetric representation keeps none of it and is not eligible), and on what fraction of instances the representation beats the evaluator's own Euclidean embedding on both distortion and MAP.

## citeseer_digraph_reconstruction
The same measurement on the Citeseer citation network (2,120-paper largest component; 5 visible and 10 hidden 400-paper subgraphs), checking that the claim holds on a second real digraph.

The paper reports the tree experiments qualitatively, so the reference file encodes those claims as pass checks. For the citation experiments, the reference file lists the paper's Table 1 values at dimension 10 (obtained in a different setup and not on this gym's scale) and the claim they support.
