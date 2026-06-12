# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-12 03:34] - core_claim
- Reproduced the paper's directed binary-tree Finsler SMACOF hierarchy result in the scheduled container.
- Verified the evaluator writes artifact-based `scores.json` and passes `python validate.py --compare`.

### [2026-06-12 03:55] - core_claim_plus
- Added and scored a second directed-tree depth setting.
- Confirmed the Finsler SMACOF implementation preserves asymmetric distances and hierarchy direction while the Euclidean SMACOF baseline fails those checks.

## Stop Justification
- Completed at milestone `core_claim_plus`.
