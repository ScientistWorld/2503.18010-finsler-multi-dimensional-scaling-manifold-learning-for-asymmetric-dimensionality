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

### [2026-11-16 23:59] - core_claim_plus
- Hardening (iteration 11): added the paper's Table 1 citation digraphs (Cora, Citeseer) as scored experiments with hidden test subgraphs. In 10D the paper's Finsler SMACOF beats the Euclidean embedding on distortion and MAP (hidden test: Cora 0.135 vs 0.235 / 0.440 vs 0.426; Citeseer 0.129 vs 0.222 / 0.492 vs 0.443); in 2D it does not.
- Audit replayed variants and shortcuts on the new experiments: better solvers rank well above seed noise, all shortcuts fail (`HARDENING.md`).

## Stop Justification
- Completed at milestone `core_claim_plus`.
