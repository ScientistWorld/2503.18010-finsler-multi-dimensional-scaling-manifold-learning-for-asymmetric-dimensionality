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

## How Methods Are Evaluated

The evaluators **run your method** on the instances themselves; they never read results you computed in advance.

1. Register the method with a card anywhere under `outputs/`, e.g. `outputs/my_method/method.json`:

   ```json
   {"name": "my_method", "command": ["python", "-m", "my_package.embed", "--some-flag"]}
   ```

2. The evaluator appends `--instances <dir> --output <dir>` and runs the command from the workspace root (no network, one CPU, at most 3600 s for all instances of a slice).
3. `<instances>/index.json` lists `{"id", "file"}` entries; each instance file (`.npz`) holds `dissimilarities` (N x N directed distances, row = source) and an integer `seed`.
4. For every instance the method writes `<output>/<id>.npz` with
   - `embedding`: finite array of shape (N, 2), and
   - `drift`: 2-vector with norm < 1 (or `alpha`: a scalar, meaning drift `(0, alpha)`).

   Everything else in the file is ignored. Predicted directed distances are `d(i -> j) = ||x_j - x_i|| + drift . (x_j - x_i)`.

`baseline/embed.py` is a complete example of the interface.

Scripts (no arguments):
- `bash scripts/evaluate.sh`: the paper's exact trees (depth 7 and depth 5), written to `scoring/scores.json`.
- `bash scripts/evaluate_train.sh`: the visible instance family (8 depth-7 and 12 depth-5 trees), written to `scoring/scores_train.json`.
- `bash scripts/evaluate_test.sh`: hidden instances from the same family (more of them), written to `scoring/scores_test.json`.

Every score file also contains `euclidean_mds`, a symmetric Euclidean SMACOF baseline that the evaluator runs from its own frozen copy.

## Metrics (per instance, averaged over the experiment's instances)

- `normalized_stress` (lower is better): `sqrt(sum (d - D)^2 / sum D^2)` over all ordered node pairs. Global reconstruction error.
- `distortion` (lower is better): mean over ordered pairs of `(d - D)^2 / D`, the distortion measure of the paper's digraph experiments. It weights short distances, so it reflects local structure.
- `symmetric_rmse` (lower is better): RMSE of `(d_ij + d_ji)/2` against `(D_ij + D_ji)/2`.
- `asymmetry_rmse` (lower is better): RMSE of `d_ij - d_ji` against `D_ij - D_ji`.
- `distance_preservation_pass` (higher is better): `clip(1 - normalized_stress / 0.10, 0, 1)`.
- `asymmetry_preservation_pass`, `hierarchy_direction_pass` (higher is better): fraction of instances meeting `asymmetry_rmse <= 0.10` and `hierarchy_alignment >= 0.80`.
- Diagnostics, reported but not weighted:
  - `map_at_10`: tie-aware MAP of the 10 nearest directed neighbours.
  - `asymmetry_correlation`.
  - `hierarchy_alignment`: signed agreement, over pairs of nodes on different levels, between the hierarchy and each node's uphill potential (total incoming minus total outgoing predicted distance). It is 0 for symmetric distances.

## Eligibility

A method is eligible only if, on **every** instance of **both** experiments, it returns a valid output (right shape, finite, drift norm < 1, command exited with status 0) and meets all of: `normalized_stress <= 0.10`, `distortion <= 0.10`, `asymmetry_rmse <= 0.10`, `hierarchy_alignment >= 0.80`. A method that is not eligible is scored as failed: every metric of every experiment takes the value of the empty (all-zero) prediction, which is worse than any eligible value.

## Baselines and Targets

The symmetric Euclidean baseline cannot represent direction. Its raw stress is about 0.48 and its asymmetry RMSE about 1.5-1.9, so it is not eligible. The paper reports the tree experiment qualitatively. `scoring/reference.json` therefore encodes those qualitative claims as pass checks and keeps the paper's quantitative digraph numbers (other datasets) under `_paper_table1_2d_reconstruction` for context only.

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

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:34] - core_claim
- Reproduced the paper's directed binary-tree Finsler SMACOF hierarchy result in the scheduled container.
- Verified the evaluator writes artifact-based `scores.json` and passes `python validate.py --compare`.

### [2026-06-12 03:55] - core_claim_plus
- Added and scored a second directed-tree depth setting.
- Confirmed the Finsler SMACOF implementation preserves asymmetric distances and hierarchy direction while the Euclidean SMACOF baseline fails those checks.

### [2026-06-12 04:20] - core_claim_plus
- Continuation audit verified the method is a real implementation of canonical Randers Finsler SMACOF.
- Fixed evaluator reusability by adding manifest-based artifact scoring for future methods.

### [2026-06-12 04:45] - core_claim_plus
- CPU job `bfea18f8-f56` completed successfully after the evaluator fix.
- Verified `scripts/reproduce.sh`, `python validate.py --compare`, and manifest-mode scoring inside the container.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:34] - core_claim
- Reproduced the paper's directed binary-tree Finsler SMACOF hierarchy result in the scheduled container.
- Verified the evaluator writes artifact-based `scores.json` and passes `python validate.py --compare`.

### [2026-06-12 03:55] - core_claim_plus
- Added and scored a second directed-tree depth setting.
- Confirmed the Finsler SMACOF implementation preserves asymmetric distances and hierarchy direction while the Euclidean SMACOF baseline fails those checks.

### [2026-06-12 04:20] - core_claim_plus
- Continuation audit verified the method is a real implementation of canonical Randers Finsler SMACOF.
- Fixed evaluator reusability by adding manifest-based artifact scoring for future methods.

### [2026-06-12 04:45] - core_claim_plus
- CPU job `bfea18f8-f56` completed successfully after the evaluator fix.
- Verified `scripts/reproduce.sh`, `python validate.py --compare`, and manifest-mode scoring inside the container.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:34] - core_claim
- Reproduced the paper's directed binary-tree Finsler SMACOF hierarchy result in the scheduled container.
- Verified the evaluator writes artifact-based `scores.json` and passes `python validate.py --compare`.

### [2026-06-12 03:55] - core_claim_plus
- Added and scored a second directed-tree depth setting.
- Confirmed the Finsler SMACOF implementation preserves asymmetric distances and hierarchy direction while the Euclidean SMACOF baseline fails those checks.

### [2026-06-12 04:20] - core_claim_plus
- Continuation audit verified the method is a real implementation of canonical Randers Finsler SMACOF.
- Fixed evaluator reusability by adding manifest-based artifact scoring for future methods.

### [2026-06-12 04:45] - core_claim_plus
- CPU job `bfea18f8-f56` completed successfully after the evaluator fix.
- Verified `scripts/reproduce.sh`, `python validate.py --compare`, and manifest-mode scoring inside the container.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 10 total (5 completed, 5 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:34] - core_claim
- Reproduced the paper's directed binary-tree Finsler SMACOF hierarchy result in the scheduled container.
- Verified the evaluator writes artifact-based `scores.json` and passes `python validate.py --compare`.

### [2026-06-12 03:55] - core_claim_plus
- Added and scored a second directed-tree depth setting.
- Confirmed the Finsler SMACOF implementation preserves asymmetric distances and hierarchy direction while the Euclidean SMACOF baseline fails those checks.

### [2026-06-12 04:20] - core_claim_plus
- Continuation audit verified the method is a real implementation of canonical Randers Finsler SMACOF.
- Fixed evaluator reusability by adding manifest-based artifact scoring for future methods.

### [2026-06-12 04:45] - core_claim_plus
- CPU job `bfea18f8-f56` completed successfully after the evaluator fix.
- Verified `scripts/reproduce.sh`, `python validate.py --compare`, and manifest-mode scoring inside the container.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 10 total (5 completed, 5 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:34] - core_claim
- Reproduced the paper's directed binary-tree Finsler SMACOF hierarchy result in the scheduled container.
- Verified the evaluator writes artifact-based `scores.json` and passes `python validate.py --compare`.

### [2026-06-12 03:55] - core_claim_plus
- Added and scored a second directed-tree depth setting.
- Confirmed the Finsler SMACOF implementation preserves asymmetric distances and hierarchy direction while the Euclidean SMACOF baseline fails those checks.

### [2026-06-12 04:20] - core_claim_plus
- Continuation audit verified the method is a real implementation of canonical Randers Finsler SMACOF.
- Fixed evaluator reusability by adding manifest-based artifact scoring for future methods.

### [2026-06-12 04:45] - core_claim_plus
- CPU job `bfea18f8-f56` completed successfully after the evaluator fix.
- Verified `scripts/reproduce.sh`, `python validate.py --compare`, and manifest-mode scoring inside the container.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 10 total (5 completed, 5 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:34] - core_claim
- Reproduced the paper's directed binary-tree Finsler SMACOF hierarchy result in the scheduled container.
- Verified the evaluator writes artifact-based `scores.json` and passes `python validate.py --compare`.

### [2026-06-12 03:55] - core_claim_plus
- Added and scored a second directed-tree depth setting.
- Confirmed the Finsler SMACOF implementation preserves asymmetric distances and hierarchy direction while the Euclidean SMACOF baseline fails those checks.

### [2026-06-12 04:20] - core_claim_plus
- Continuation audit verified the method is a real implementation of canonical Randers Finsler SMACOF.
- Fixed evaluator reusability by adding manifest-based artifact scoring for future methods.

### [2026-06-12 04:45] - core_claim_plus
- CPU job `bfea18f8-f56` completed successfully after the evaluator fix.
- Verified `scripts/reproduce.sh`, `python validate.py --compare`, and manifest-mode scoring inside the container.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 10 total (5 completed, 5 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:34] - core_claim
- Reproduced the paper's directed binary-tree Finsler SMACOF hierarchy result in the scheduled container.
- Verified the evaluator writes artifact-based `scores.json` and passes `python validate.py --compare`.

### [2026-06-12 03:55] - core_claim_plus
- Added and scored a second directed-tree depth setting.
- Confirmed the Finsler SMACOF implementation preserves asymmetric distances and hierarchy direction while the Euclidean SMACOF baseline fails those checks.

### [2026-06-12 04:20] - core_claim_plus
- Continuation audit verified the method is a real implementation of canonical Randers Finsler SMACOF.
- Fixed evaluator reusability by adding manifest-based artifact scoring for future methods.

### [2026-06-12 04:45] - core_claim_plus
- CPU job `bfea18f8-f56` completed successfully after the evaluator fix.
- Verified `scripts/reproduce.sh`, `python validate.py --compare`, and manifest-mode scoring inside the container.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 11 total (6 completed, 5 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:34] - core_claim
- Reproduced the paper's directed binary-tree Finsler SMACOF hierarchy result in the scheduled container.
- Verified the evaluator writes artifact-based `scores.json` and passes `python validate.py --compare`.

### [2026-06-12 03:55] - core_claim_plus
- Added and scored a second directed-tree depth setting.
- Confirmed the Finsler SMACOF implementation preserves asymmetric distances and hierarchy direction while the Euclidean SMACOF baseline fails those checks.

### [2026-06-12 04:20] - core_claim_plus
- Continuation audit verified the method is a real implementation of canonical Randers Finsler SMACOF.
- Fixed evaluator reusability by adding manifest-based artifact scoring for future methods.

### [2026-06-12 04:45] - core_claim_plus
- CPU job `bfea18f8-f56` completed successfully after the evaluator fix.
- Verified `scripts/reproduce.sh`, `python validate.py --compare`, and manifest-mode scoring inside the container.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 32m | **GPU**: 0.0h
- **Jobs**: 12 total (7 completed, 5 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-12 03:34] - core_claim
- Reproduced the paper's directed binary-tree Finsler SMACOF hierarchy result in the scheduled container.
- Verified the evaluator writes artifact-based `scores.json` and passes `python validate.py --compare`.

### [2026-06-12 03:55] - core_claim_plus
- Added and scored a second directed-tree depth setting.
- Confirmed the Finsler SMACOF implementation preserves asymmetric distances and hierarchy direction while the Euclidean SMACOF baseline fails those checks.

### [2026-06-12 04:20] - core_claim_plus
- Continuation audit verified the method is a real implementation of canonical Randers Finsler SMACOF.
- Fixed evaluator reusability by adding manifest-based artifact scoring for future methods.

### [2026-06-12 04:45] - core_claim_plus
- CPU job `bfea18f8-f56` completed successfully after the evaluator fix.
- Verified `scripts/reproduce.sh`, `python validate.py --compare`, and manifest-mode scoring inside the container.

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
