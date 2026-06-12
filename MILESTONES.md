# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

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

## Stop Justification
- Completed at milestone `core_claim_plus`. The packaged environment demonstrates the core Finsler MDS claim on the main hierarchy benchmark and one additional graph size with a repaired scoring contract.
