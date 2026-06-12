# Reproduction Milestones

**Current: core_claim_plus**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

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

## Stop Justification
- Stopped at `core_claim_plus`: the reusable environment reproduces the paper's Finsler SMACOF computation and core asymmetric-distance claim on the Appendix D.1.1 tree benchmark plus an additional tree size. Higher milestones would require implementing and running the larger directed-graph Table 1 suite.
