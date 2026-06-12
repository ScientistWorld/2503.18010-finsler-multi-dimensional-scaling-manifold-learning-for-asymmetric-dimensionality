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

## Current Submission
- Running a CPU test job after the evaluator fix to verify `scripts/reproduce.sh`, `validate.py --compare`, and manifest-mode scoring inside the container.
