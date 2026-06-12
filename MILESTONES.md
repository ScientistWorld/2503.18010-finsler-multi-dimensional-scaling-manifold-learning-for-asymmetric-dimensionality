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
- The environment is complete at `core_claim_plus`: it reproduces the paper's core directed-tree Finsler SMACOF claim and an additional tree-depth setting, with a reusable artifact-based evaluator and packaged scripts.
- Remaining work toward `majority` would require implementing the larger neural directed-graph benchmark suite from Table 1; the current deliverable is a validated research gym for the central Finsler asymmetric-embedding claim.
