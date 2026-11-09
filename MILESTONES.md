# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

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

## Stop Justification
- Completed at milestone `core_claim_plus`.

### [2026-11-09] - core_claim_plus (hardening)
- The test now runs each registered method on hidden per-level-jittered trees. Scores are recomputed from validated 2D embeddings, and an eligibility gate turns invalid outputs or missed claim thresholds into failure scores.
- The reference still reproduces the paper's claim on every slice: hidden-test stress 0.023 / 0.027 vs a Euclidean baseline at 0.48 / 0.47.
- A corrected-majorization SMACOF beats the reference by 177× its seed range. All red-team shortcuts now fail. See HARDENING.md.
