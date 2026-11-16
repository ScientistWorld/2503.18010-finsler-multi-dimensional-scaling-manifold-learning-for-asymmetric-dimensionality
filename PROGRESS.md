# Progress

## What Works

- Paper analysis is complete for the minimum SMACOF reproduction path.
- The official code release is vendored under `method/upstream_finsler_mds`.
- Local tracked code now implements the Appendix D.1.1 directed binary-tree benchmark, the paper's canonical Randers Finsler SMACOF update, a Euclidean SMACOF baseline, and an evaluator that reads saved artifacts.
- CPU job `215fb11e-2bf` ran `scripts/reproduce.sh` successfully inside the container and `python validate.py --compare` passed.
- CPU job `fd4957cf-552` successfully regenerated `scoring/scores.json` with both the original depth-7 tree benchmark and an additional depth-5 tree benchmark.
- Final local validation on 2026-06-12 confirmed `reference.json`, `scores.json`, import separation, artifact-based evaluation, and shared-path hygiene all pass.
- Follow-up audit found that the previous `reference.json` used invented continuous targets for the qualitative Figure 5 tree result. The score contract has been repaired so `scores.json` contains binary checks of paper-stated qualitative claims, while continuous diagnostics are written to `scoring/details.json`.
- CPU job `7d307bfa-ccd` reran the full reproduction under the repaired scoring contract and `python validate.py --compare` passed.
- A final packaging pass moved the shared Isomap/classical-scaling initializer from `method/` to `data/`, so the Euclidean baseline no longer imports the proposed-method package. The optional unrun Cora reference entry now uses method labels matching the paper's neural directed-graph embedding setup.
- Continuation audit confirmed the Finsler SMACOF implementation is faithful, then fixed a reusable-evaluation gap: `eval/tree_metrics.py` now supports a manifest of artifact paths and artifacts with precomputed `predicted_distances`, while preserving the default reproduction interface.
- CPU job `bfea18f8-f56` reran the full reproduction after the reusable-evaluator fix; `scripts/reproduce.sh`, `python validate.py --compare`, and manifest-mode scoring all completed successfully.
- Final audit fixed `scripts/run.sh` so submitted runs refresh canonical `scoring/scores.json` before the split train/test score files.

## Results

- Citation digraphs (iteration 11; official files from job `b4c96983-d30`, hidden test slice, 10D, reference vs the evaluator's Euclidean SMACOF):
  - `cora_digraph_reconstruction`: distortion `0.135` vs `0.235`, MAP `0.440` vs `0.426`, stress `0.109` vs `0.152`, asymmetry error `0.40` vs `1.00`; Table 1 claim (both better) on 7/10 subgraphs.
  - `citeseer_digraph_reconstruction`: distortion `0.129` vs `0.222`, MAP `0.492` vs `0.443`, stress `0.086` vs `0.122`, asymmetry error `0.41` vs `1.00`; claim on 10/10 subgraphs.
  - Whole graphs (`scores.json`): Cora distortion `0.170` vs `0.247`, MAP `0.406` vs `0.384`; Citeseer `0.160` vs `0.226`, `0.476` vs `0.424`.
  - In 2D the claim does not reproduce (see the iteration-11 note above).
- Hardening audit of the citation experiments: see `HARDENING.md` (better solvers rank +2.07 to +2.59 above the reference in the eligible pool, all shortcuts fail).

- `directed_tree_reconstruction` (depth 7):
  - Euclidean SMACOF baseline: normalized stress `0.5028`, MAP@10 `0.2669`, asymmetry RMSE `1.8698`, symmetric RMSE `0.5379`, hierarchy alignment `0.0082`.
  - Finsler MDS: normalized stress `0.0222`, MAP@10 `0.2679`, asymmetry RMSE `0.0188`, symmetric RMSE `0.0466`, hierarchy alignment `0.9249`.
- `directed_tree_depth5_reconstruction`:
  - Euclidean SMACOF baseline: normalized stress `0.4949`, MAP@10 `0.6148`, asymmetry RMSE `1.6849`, symmetric RMSE `0.4496`, hierarchy alignment `0.0231`.
  - Finsler MDS: normalized stress `0.0254`, MAP@10 `0.5894`, asymmetry RMSE `0.0197`, symmetric RMSE `0.0481`, hierarchy alignment `0.9222`.
- These continuous diagnostics support the paper's central claim that a canonical Randers Finsler geometry can represent directed asymmetric distances that a symmetric Euclidean embedding cannot. The official score file now records the corresponding paper-claim pass/fail checks.
- Official repaired scores: Euclidean MDS gets `0` on the qualitative distance, asymmetry, and hierarchy checks; Finsler MDS clears the asymmetry and hierarchy checks and receives graded distance-preservation scores of `0.7785` on depth 7 and `0.7458` on depth 5. Continuous stress, ranking, asymmetry, symmetry, and hierarchy diagnostics are also included in `scores.json`.
- Current milestone: `core_claim_plus`, because the repaired scoring run confirms the core directed-tree result and repeats it on an additional tree size.
- Final validation job `bfea18f8-f56` preserved the same official repaired scores and passed all structure checks.

## Remaining

- Full reproduction of Table 1 graph datasets remains future work; the packaged environment currently targets the paper's Appendix D.1.1 hierarchy benchmark plus one scaled setting.
- To reach `majority` or higher, implement the paper's directed-graph suite for Cora/Citeseer/Pubmed/Airports/Communities and add the neural embedding variants reported in Table 1.

## Issues

- The login-node Python environment does not have NumPy installed, so numerical experiments are run through scheduled jobs. Dependencies are declared in `environment/setup.sh` for the scheduled container job.
- The tree visualization experiment in the paper is reported qualitatively rather than with a numeric table. `reference.json` now encodes only qualitative pass/fail claim checks for the tree task and keeps the paper Table 1 Cora numbers as an unrun reference target.
- CPU job `aa29f577-e55` failed before scoring because `scripts/evaluate.sh` still passed removed `--finsler` and `--euclidean` options. The script now calls the evaluator's default experiment suite.
- Initial tree reference stress targets were benchmark-scale values rather than paper-reported numbers. This audit replaced them with binary qualitative claim checks and moved continuous diagnostics out of `scores.json`.
- The continuation audit found that the previous evaluator was independent from `method/` but too hard-coded for future methods. This has been fixed with a JSON manifest interface and a `predicted_distances` artifact path; `scripts/run.sh` now tests that interface in the container.
- The final CPU test completed successfully. Remaining work is expansion to the paper's larger neural graph benchmarks, not a known correctness bug in the packaged tree reproduction.
- A side-agent compliance check found that `scripts/run.sh` refreshed only split scores; it now calls `scripts/evaluate.sh` as well.

- Citation-digraph experiments (Cora, Citeseer; paper Table 1), iteration 11. Calibration job `6fb65fac-dc7` (visible subgraphs and whole graphs, reverse traversal at cost 3) compared the reference Finsler SMACOF (alpha 0.5) with Euclidean SMACOF in the paper's Table 1 dimensions. In 2D the claim does **not** reproduce: on the visible subgraphs the reference loses to Euclidean on distortion (Cora 1.27 vs 0.79, Citeseer 1.58 vs 0.89) and MAP (0.089 vs 0.130, 0.135 vs 0.178), and its asymmetry error exceeds 1 (worse than a symmetric embedding); smaller drifts and exact/weighted/L-BFGS variants do not change this. In 5D the reference wins on Citeseer but not on Cora MAP (0.266 vs 0.282). In 10D it wins on both graphs (Cora distortion 0.134 vs 0.214, MAP 0.408 vs 0.390; Citeseer 0.136 vs 0.219, 0.509 vs 0.480; whole graphs likewise). The rule fixed before the calibration (smallest Table 1 dimension where the reference beats Euclidean on both distortion and MAP on the visible slice) selects 10D, so the citation experiments embed in 10D; the trees stay 2D (Figure 5).

## Deviations from Paper

<!-- Do not edit this section unless you deviated from the paper's original setup.
     For each deviation, document: what the paper used, what you substituted, and why.

     ACCEPTABLE deviations (data/resource/scale substitutions — milestones still valid):
       - Replaced a proprietary dataset with a public one
       - Used a smaller model of the same architecture
       - Scaled down training (fewer epochs, smaller batch)
       - Used a different library for the same algorithm

     UNACCEPTABLE deviations (algorithm substitutions — milestones above 'none' are void):
       - Replaced the paper's algorithm with a fundamentally different one
       - Wrote a generic model trained to approximate the method's outputs
         instead of implementing the method's computational steps

     The test: could someone reading your code identify which algorithm from
     the paper you implemented? If not, you replaced the method. -->

- The first executable reproduction uses the paper's directed binary-tree visualization task rather than the full neural directed-graph experiments. This preserves the paper's Finsler SMACOF algorithm and Appendix D.1.1 procedure while avoiding large graph representation-learning runs before the core method is validated.
- The evaluator adds explicit numeric metrics for the qualitative Figure 5 tree result: normalized stress, MAP@10, asymmetry RMSE, symmetric RMSE, and hierarchy alignment. These quantify the same reconstruction and hierarchy-preservation claim shown visually in the paper.
- Citation-digraph experiments: the paper learns Table 1 embeddings with an MLP (Riemannian SGD adapted to the Randers metric) on directed shortest-path distances and does not say how unreachable pairs are handled. The gym uses the paper's Finsler SMACOF (Sec. 6.1, same Finsler MDS problem) as the reference, makes all directed distances finite with the paper's own tree convention (reverse traversal costs 3x), embeds in dimension 10 (a Table 1 dimension; 2D does not reproduce, see above) and uses 400-node subgraphs for the train/test slices (whole largest components for scores.json).
