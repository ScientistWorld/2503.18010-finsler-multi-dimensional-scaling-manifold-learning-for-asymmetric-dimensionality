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

---

# Problem Definition

## Research Question

Given a set of objects with pairwise dissimilarities that can be asymmetric, find a low-dimensional embedding whose directed pairwise distances preserve those dissimilarities. The benchmark focuses on directed graph distances, where the shortest path from node `i` to node `j` may differ from the reverse path.

## Why It Matters

Many relationships are directional: hierarchy, road or current travel time, citations, links, and information flow. A useful low-dimensional representation should preserve both proximity and directionality so that visualization, reconstruction, and link analysis do not discard the signal carried by asymmetry.

## Success Criteria

A successful system produces low-dimensional coordinates whose induced directed distances have low reconstruction error against the target directed shortest-path distances, preserve nearest-neighbour rankings, and retain the asymmetric component that a symmetric embedding cannot represent.

---

# Paper's Method

## Key Contribution

The paper generalizes multidimensional scaling to asymmetric dissimilarities by embedding data in a canonical Finsler, specifically Randers, space instead of a symmetric Riemannian/Euclidean space.

## Approach

The canonical embedding space keeps straight-line geodesics but assigns direction-dependent lengths:
`d(x, y) = ||y - x||_2 + omega^T (y - x)`, with a constant drift vector `omega` of norm below one. The paper minimizes a Finsler stress objective and derives a SMACOF-style majorization update. For larger examples it uses a stabilized linear solve based on multiplying the update equation by `V^T` and solving with GMRES.

The paper also presents a neural representation-learning version for larger directed graphs, using the same canonical distance in a Fermi-Dirac decoder for link prediction.

## Main Claims

Primary claim: low-dimensional embeddings with directed distances preserve asymmetric dissimilarities better than Euclidean embeddings.

Secondary claims: the same framework supports intuitive visualizations of asymmetric manifolds, current maps, and graph hierarchies; it is compatible with robust MDS weighting schemes; and the representation-learning variant improves directed graph reconstruction and link prediction.

---

# Evaluation

## Metrics

- `normalized_stress`: square-root stress normalized by the squared target dissimilarities. Lower is better.
- `map_at_10`: mean average precision of the top-10 nearest directed neighbours under the predicted distances. Higher is better.
- `asymmetry_rmse`: RMSE between target and predicted antisymmetric distance components, `D_ij - D_ji`. Lower is better.
- `symmetric_rmse`: RMSE between target and predicted symmetric distance components. Lower is better.
- `hierarchy_alignment`: absolute Spearman correlation between the embedding's final coordinate and tree depth on the hierarchy benchmark. Higher is better.
- `distance_preservation_pass`: binary Figure 5 claim check derived from normalized stress. Higher is better.
- `asymmetry_preservation_pass`: binary Figure 5 claim check derived from asymmetry RMSE. Higher is better.
- `hierarchy_direction_pass`: binary Figure 5 claim check derived from hierarchy alignment. Higher is better.

## Baselines and Targets

The primary baseline is a Euclidean low-dimensional embedding optimized against the same directed dissimilarity matrix. The paper's Table 1 reports that on Cora with 2D embeddings, Euclidean reconstruction has distortion `0.426` and MAP `0.574`, while the proposed directed-distance embedding has distortion `0.127` and MAP `0.894`.

## Evaluation Protocol

For the packaged reproduction, run the fixed directed binary-tree benchmark from the paper's visualization experiments. Generate the graph, compute all-pairs directed shortest-path dissimilarities, run the baseline and proposed method with the same seed and embedding dimension, then evaluate only the saved output artifacts with `scripts/evaluate.sh`.

The paper reports the tree result qualitatively rather than as a numeric table. To avoid inventing continuous paper values, `scoring/scores.json` reports binary checks for the qualitative Figure 5 claims, while `scoring/details.json` records the continuous normalized stress, MAP@10, asymmetry RMSE, symmetric RMSE, and hierarchy alignment diagnostics.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 14m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:12] - none
- Read the paper markdown and extracted the core Finsler MDS formulation, canonical Randers distance, Finsler SMACOF update, and Appendix D.1.1 directed-tree experiment.
- Cloned the official code release into `method/upstream_finsler_mds` for reference.
- Implemented the directed binary-tree data generator, Finsler SMACOF method, Euclidean SMACOF baseline, and method-independent evaluation code.
- Static validation passes; first CPU test job is being submitted to execute the reproduction inside the container.

### [2026-06-12 03:31] - core_claim
- CPU job `215fb11e-2bf` completed successfully inside the container.
- The paper's Finsler SMACOF algorithm ran end-to-end on the Appendix D.1.1 directed-tree benchmark.
- Finsler MDS reduced normalized stress from `0.5028` to `0.0222` versus Euclidean SMACOF and reduced asymmetry RMSE from `1.8698` to `0.0188`, supporting the core asymmetric-embedding claim.
- Added a second depth-5 tree setting for the next CPU job to test the same claim beyond one graph size.

### [2026-06-12 03:42] - core_claim_plus
- CPU job `fd4957cf-552` completed successfully inside the container and `python validate.py --compare` passed.
- On the added depth-5 tree setting, Finsler MDS reduced normalized stress from `0.4949` to `0.0254` and asymmetry RMSE from `1.6849` to `0.0197` versus Euclidean SMACOF.
- The same asymmetric reconstruction advantage now holds on two directed-tree sizes, so the reproduction reaches `core_claim_plus`.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:27] - method_runs
- Implemented the paper's canonical Randers Finsler SMACOF update for asymmetric dissimilarity matrices.
- Added the directed binary-tree benchmark and Euclidean SMACOF baseline.
- Verified the method and baseline execute end-to-end through scheduled CPU jobs.

### [2026-06-12 03:27] - core_claim
- Reproduced the paper's Figure 5 directed hierarchy claim on a depth-7 directed binary tree.
- Finsler MDS achieved much lower normalized stress and asymmetry RMSE than Euclidean MDS while aligning the learned vertical coordinate with graph hierarchy.
- Evaluation reads saved artifacts from `outputs/` and does not import from `method/`.

### [2026-06-12 03:28] - core_claim_plus
- Added a second depth-5 directed-tree setting using the same algorithm and evaluator.
- The Finsler embedding again reconstructed asymmetric distances far better than the symmetric Euclidean baseline.
- `python validate.py --compare` passes for scoring structure, separation rules, and artifact-based evaluation.

### [2026-06-12 04:00] - core_claim
- Follow-up audit confirmed the implementation is the paper's Finsler SMACOF algorithm rather than a surrogate.
- Found and fixed a scoring-integrity issue: continuous numeric targets for qualitative Figure 5 claims were moved out of `reference.json`.
- A CPU rerun is being submitted to regenerate `scores.json` with the repaired qualitative claim-check evaluator.

### [2026-06-12 04:02] - core_claim_plus
- CPU job `7d307bfa-ccd` reran `scripts/reproduce.sh` and `python validate.py --compare` successfully.
- The repaired evaluator matches the qualitative Figure 5 reference checks: Euclidean MDS fails distance/asymmetry/hierarchy checks, while Finsler MDS passes all checks on both depth-7 and depth-5 tree settings.
- Continuous diagnostics are preserved in `scoring/details.json` for auditability without treating non-paper values as reference numbers.

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
