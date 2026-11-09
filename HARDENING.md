# Hardening Report

All "after" numbers come from CPU job `9b3a14c7-d19` (exit 0, 44 min). It:
- scored the reference on the hardened paper, train and test slices;
- replayed every variant and shortcut as a real method card through the unchanged `bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh` (no arguments);
- ran three file-modifying shortcuts in isolation;
- regenerated the official score files from the reference card alone. The second run was byte-identical to the first (`cmp`), and `python validate.py` passed.

The harness is [redteam/hardening/audit.py](redteam/hardening/audit.py), with variants in [variants.py](redteam/hardening/variants.py) and shortcuts in [shortcuts.py](redteam/hardening/shortcuts.py). Raw results, ranks and pools are in [results/summary.json](redteam/hardening/results/summary.json) and [results/results_table.md](redteam/hardening/results/results_table.md). "Before" numbers are quoted from `REDTEAM.md`.

## What changed in the gym

1. **The test runs the method on hidden instances.** Finsler MDS needs no training. Before, the test scored saved artifacts on the same two public trees as train (only a different half of the source rows). Now:
   - A method registers a command in `outputs/<name>/method.json`.
   - The evaluators run it on directed trees and read back only an N×2 `embedding` plus a `drift` vector (norm < 1).
   - The test trees are 12 depth-7 and 20 depth-5 trees. They have the paper's topology, per-level weights drawn around the paper's 0.5 / 1.5 / 0.1, and shuffled node labels. Their seeds exist only in `eval/test/evaluate.py`.
   - Train uses the same family with 8 + 12 visible seeds. `scores.json` uses the exact Appendix D.1.1 trees.
2. **Every distance and metric is recomputed from the embedding.** `predicted_distances`, target matrices, depths, score fields and flags in method outputs are ignored.
3. **Eligibility gate.** The method fails if any instance has an invalid output or misses a claim threshold:
   - Invalid output: missing file, wrong shape, non-finite values, drift norm ≥ 1, a non-zero exit, or a bad card.
   - Thresholds: stress ≤ 0.10, distortion ≤ 0.10, asymmetry RMSE ≤ 0.10, hierarchy alignment ≥ 0.80.

   A failed method gets, on every metric of both experiments, the value of the empty (all-zero) prediction. That value is worse than any eligible value on every metric, so a failed method can never outrank a solving one, whatever the coefficients or normalisation.
4. **Fixed comparison side.**
   - Each evaluator has its own copy of the generator and its own frozen Euclidean SMACOF baseline (`euclidean_mds`).
   - The test no longer imports `data/` or reads `scoring/scores_train.json`.
   - Thresholds, instance counts and seeds are constants in `eval/`.
5. **Metrics.**
   - Added `distortion`, the paper's Table 1 distortion: mean of (d − D)²/D over ordered pairs.
   - `map_at_10` is now tie-aware and `hierarchy_alignment` is now a sign-aware concordance computed from the predicted distances. Both have weight 0 (see the headroom check).
6. **Determinism.** The Isomap initialisation used ARPACK with a random start vector. It now uses a dense eigendecomposition with a fixed sign convention. The red team's stored-reference/artifact mismatch is gone: reruns reproduce the score files byte for byte.

## Do better results now score higher?

**Yes, by far more than the noise.** The reference was rerun with five seed offsets. Its per-metric spread on the test is about 1e-6 to 1e-5 in stress and 1e-5 in asymmetry RMSE. Its rank range is 0.0105 in a pool of eligible attempts, 0.00008 in the genuine pool and 0.00007 in the all-attempts pool.

Rank = Σ experiment weight × |coefficient| × improvement over the reference ÷ population standard deviation across the pool. Pools:
- **E**: the 15 eligible genuine attempts.
- **G**: E plus the failed weaker variants, the Euclidean submission and the level-collapse probe.
- **A**: every tested submission.

| Variant | Before (REDTEAM, pool Q / A) | After: rank E / G / A | Test stress d7 / d5 | Test distortion d7 / d5 |
|---|---:|---:|---|---|
| Reference (finsler_mds) | 0 / 0 | 0 / 0 / 0 | 0.022915 / 0.027025 | 0.011310 / 0.011823 |
| Reference, 4 other seeds | spread 0.52 / 0.024 | −0.0105…−0.0013 (range 0.0105) | 0.022916–0.022918 / 0.027025–0.027037 | — |
| **Corrected-majorization SMACOF** (new, 3 seeds) | — | **+1.862…+1.873** / +0.012 / +0.011 | 0.021915 / 0.025961 | 0.011073 / 0.011498 |
| L-BFGS refinement (3 seeds) | +0.126 / +0.031 (Q gain < Q noise) | **+1.867…+1.879** / +0.012 / +0.011 | 0.021911 / 0.025960 | 0.011071 / 0.011497 |
| Longer SMACOF (2000 it., eps 1e-10) | −0.369 / −0.001 | +0.006 / 0.000 / 0.000 | 0.022911 / 0.027022 | 0.011307 / 0.011822 |
| Multi-scale refinement (stress + distortion) | — | −0.51 / −0.024 / −0.020 | 0.024824 / 0.029075 | **0.009168 / 0.009468** |
| 10 SMACOF iterations | — | failed (asym RMSE 0.13–0.14): −13.6 / −7.6 / −6.6 | raw 0.044 | — |
| 5 SMACOF iterations | −2.37 / −1.78 | failed (asym RMSE 0.30–0.34) | raw 0.093 | — |
| Drift 0.3 instead of 0.5 | — | failed (asym RMSE 0.68–0.74) | raw 0.22 | — |
| Euclidean (no direction) | −10.74 / −6.16 | failed (raw stress 0.48, asym 1.5–1.6) | — | — |

- **Paired comparisons.** Every method saw the same instances and seeds. In pool E the corrected-majorization variant beat the reference by +1.8625, +1.8727 and +1.8636 for seed offsets 0, 1 and 2, and L-BFGS by +1.8683, +1.8791 and +1.8666. The smallest paired gain is 177× the reference's 5-seed range. The gains hold on train too: stress 0.02425 / 0.02623 vs the reference's 0.02516 / 0.02739.
- **The corrected majorization is a real improvement, not just more search.** The released Finsler SMACOF (copied from the authors' code into `method/finsler_smacof.py`) differs from Proposition 2: it uses (1+α) instead of (1+α²) on the drift axis and the Randers distance instead of the Euclidean norm inside B(X). Its fixed points are therefore not stationary points of the Finsler stress. A pure-Python check on the depth-3 and depth-4 paper trees showed the released update stopping at |∇σ| = 1.3 and 3.8, and not decreasing monotonically. The exact majorization `X ← (B̃(X)X − C/2)·diag(1, 1+α²)⁻¹/N` decreased monotonically to |∇σ| ≈ 3e-7, with 6.5–7% lower stress. This is why 2,000 more iterations of the released update gain nothing while a correct solver gains ~4%. **The reference keeps the authors' released update.** The task forbids changing the paper's method, and the corrected update is left as headroom for future agents.
- **Trade-offs score as trade-offs.** The better variants cut stress (−4%), distortion (−2 to −3%) and symmetric RMSE (−6%), but raise asymmetry RMSE from 0.024 to 0.028; the net is clearly positive. The multi-scale variant cuts distortion by 19% but raises stress by 8% and asymmetry RMSE by 60%, so it ranks below the reference. That is a trade-off rather than a strictly better result.
- **Weaker methods all fail.** No weaker method scores between the floor and the reference: short SMACOF runs fit the symmetric part before the directional part, and fail the asymmetry gate.

## Shortcuts: fix, and score before and after

"Before" ranks are the red team's (Q+ / A pools against the stored reference). "After" ranks are in pools E / G / A. A failed submission gets the floor in every pool.

| Shortcut | Before | Fix | After |
|---|---|---|---|
| **answer_copy**: input matrix as `predicted_distances` | **+1.96 / +1.20**; stress 0, MAP 1 | distances computed only from the 2D embedding | failed (stress 1 on the zero embedding): −13.6 / −7.3 / −6.6 |
| **hierarchy_field**: reference distances plus an unrelated depth field | **+0.024 / +0.100** | one embedding gives both the distances and the hierarchy | scored as the depth-field embedding: stress up to 0.124, failed −13.6 / −7.3 / −6.6 |
| **fixed_generator**: replace `data/directed_tree.py` | **+1.96 / +1.20**; stress 0 | test has its own generator copy and hidden seeds | patched generator had no effect: reference row byte-identical to the batch; generator-based coordinates failed |
| **task_failure**: copy the symmetric part, attenuate direction | **+0.711 / +0.161** with both passes 0 | eligibility gate | drift 0.38 on the reference embedding gives asym RMSE 0.37 → failed: floor |
| move_baseline: zero the public baseline | −0.119 / −0.0003 (tie) | frozen baseline inside the evaluator | sabotaged `baseline/` has no effect; `euclidean_mds` row byte-identical |
| score_fields: perfect scores and flags in the artifact and in `scores_test.json` | −10.6 / −8.8 | only coordinates and drift read; score file rewritten | ignored, failed; forged `scores_test.json` overwritten |
| forged_targets | −10.6 / −8.8 | targets never read from the method | ignored, failed |
| impossible_diagonal (−inf self-distances) | −0.119 / −0.0003 (tie) | diagonal computed by evaluator | extra field ignored; identical to the reference (it submits the reference embedding), so it gains nothing |
| nan / inf / empty | no finite rank; evaluator exited 0 with NaN or missing rows | explicit validation | rejected as invalid → failed with finite floor scores |
| train_filter: empty `scores_train.json` | test scored nothing | test discovers cards itself | all methods still scored |
| New: 3-D embedding (more capacity) | — | shape check | rejected, failed |
| New: drift norm ≥ 1 (not a metric) | — | norm check | rejected, failed |
| New: missing one instance, crash after writing, malformed card | — | per-instance and exit-status checks | failed |
| New: level-collapse probe (all nodes on one vertical line at their potential) | — | weight moved off the saturated metrics; stress and distortion gates | exact on asymmetry, MAP and hierarchy, but stress above 0.10 on 15 of 32 hidden trees → failed. Even if it were eligible, its raw metrics rank **−10.1** in pool E (stress 0.09, distortion 0.05, symmetric RMSE 0.18), so the weights, not only the gate, keep it far below |

## Shortcuts and limitations left open

- **Selecting among many cards.** The hidden test scores every card on disk. An attempt that registers many variants could pick the best one by its hidden score. The grading protocol should rank one designated method per attempt.
- **Test-time search is part of the method.** MDS is transductive: a method necessarily receives the target matrix and can evaluate its own stress on a hidden instance. Restarts or longer optimisation at test time count as the method, bounded only by the 3600 s per-slice timeout. That timeout is a failure guard and not a metric, but it does depend on wall-clock time.
- **The method runs in the same file system as `eval/test/`.** It could read the hidden seeds. That gives it the level labels, which can already be derived from `D` itself (the asymmetric potential), and nothing that changes a score: ground truth stays in the evaluator's memory. Per the task, there are no defences against leftover processes.
- **Threshold cliffs.** A method just over a threshold on one instance drops to the floor. This is intended ("a failed constraint scores as failed"), but small changes near a threshold are discontinuous.
- **Standard-deviation normalisation.** Normalising by the std across attempts amplifies tiny differences on near-constant metrics. MAP@10 and hierarchy alignment are 1.0 for essentially every eligible method, so they now have weight 0. They are still gated through `hierarchy_direction_pass`.
- **The paper's MAP claim is not covered.** On these trees every target distance is tied within a level, so tie-aware MAP@10 is 1.0 for the reference and for the degenerate level-collapse alike. Graph-neighbour MAP even favours the Euclidean baseline (0.98 vs 0.97 on the exact depth-7 tree). The paper's MAP claim concerns real digraphs (Table 1), which this gym does not include; those numbers are in `reference.json` under `_paper_table1_2d_reconstruction`.
- **Distortion normalisation.** The paper's normalisation of its distortion measure is ambiguous (C(N,2) vs ordered pairs); the gym uses the mean over ordered pairs.
- **`validate.py` warnings.** It prints two soft warnings ("eval looks detached") because the evaluators run method commands rather than opening files with literal `outputs/` paths. Method code is exercised on every scored number.

## Reference score before and after

| | Before (stored `scores_test.json`: half of the rows of the exact trees) | After: hidden family (`scores_test.json`) | After: exact paper trees (`scores.json`) | After: visible family (`scores_train.json`) |
|---|---|---|---|---|
| normalized_stress d7 / d5 | 0.022160 / 0.026061 | 0.022915 / 0.027025 | 0.0222 / 0.0254 | 0.0252 / 0.0274 |
| distortion d7 / d5 | — | 0.011310 / 0.011823 | 0.0116 / 0.0112 | 0.0127 / 0.0119 |
| asymmetry_rmse d7 / d5 | 0.018859 / 0.019376 | 0.023598 / 0.024071 | 0.0189 / 0.0199 | 0.0262 / 0.0256 |
| symmetric_rmse d7 / d5 | 0.046511 / 0.048785 | 0.043609 / 0.046991 | 0.0467 / 0.0484 | 0.0477 / 0.0475 |
| distance_preservation_pass d7 / d5 | 0.778 / 0.739 | 0.771 / 0.730 | 0.778 / 0.746 | 0.748 / 0.726 |
| asymmetry / hierarchy passes | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| hierarchy_alignment | 0.924 / 0.919 (abs Spearman, capped at ≈ 0.92 by level ties) | 1.0 / 1.0 (concordance, diagnostic) | 1.0 / 1.0 | 1.0 / 1.0 |
| map_at_10 | 0.292 / 0.575 (index-order tie-breaking) | 1.0 / 1.0 (tie-aware, diagnostic) | 1.0 / 1.0 | 0.9999 / 0.9997 |

- The paper's claim holds in the paper's direction on every slice. The frozen Euclidean baseline has raw stress 0.50 / 0.49 on the paper trees and 0.48 / 0.47 on the hidden ones, with asymmetry RMSE 1.5–1.9. It fails the gate and is reported at the floor; its raw stress is about 20× the reference's.
- On its worst hidden instance the reference still clears every gate by ≥ 3×: stress 0.034, distortion 0.015, asymmetry RMSE 0.029, hierarchy 1.0.
- The paper's encoded results in `scoring/reference.json` (qualitative pass checks) are unchanged.

## Weights

Experiment weights are unchanged (0.6 depth 7, 0.4 depth 5). Metric coefficients changed:

| Metric | Before | After | Why |
|---|---:|---:|---|
| normalized_stress | 1.0 | 1.0 | The Finsler stress is the quantity the method optimises. |
| distortion | — | 1.0 | The paper's own reconstruction measure (Table 1). It catches degenerate embeddings that stress alone under-penalises: level-collapse distortion is 0.039 vs 0.012 for the reference on the paper tree. |
| asymmetry_rmse | 0.8 | 0.2 | Now a gated constraint. The trivial level-collapse embedding hits its bound (0) exactly, so a large continuous weight rewarded collapsing structure. |
| symmetric_rmse | 0.4 | 0.3 | Guard on the part a direction-only fit cannot reduce. |
| hierarchy_alignment | 0.5 | 0 | Saturated at its bound (1.0) for every eligible method; enforced by the gate and `hierarchy_direction_pass`. |
| map_at_10 | 0.4 | 0 | Saturated (1.0) by level ties for every level-preserving embedding, including degenerate ones. |
| distance_preservation_pass / asymmetry_preservation_pass / hierarchy_direction_pass | 0.3 / 0.3 / 0.2 | unchanged | Claim checks; the last two are the eligibility indicators. |

## Headroom check

- **At the bound for the reference:**
  - `hierarchy_alignment` (1.0) and `map_at_10` (1.0) are now weight-0 diagnostics.
  - `asymmetry_preservation_pass` and `hierarchy_direction_pass` (1.0) are binary constraints that every eligible method meets.
- **Not at the bound:** `normalized_stress` (0.023 / 0.027 vs 0), `distortion` (0.011 / 0.012 vs 0), `asymmetry_rmse`, `symmetric_rmse`, `distance_preservation_pass` (0.77 / 0.73 vs 1).
- **Headroom is real.**
  - Two independent better solvers beat the reference by 177× its seed range without reaching any bound.
  - The one simple variant that hits several bounds (level-collapse) fails the stress gate, and would rank −10.1 even if it were eligible.
  - The Euclidean baseline sits far below.
  - The improvement comes from where the paper's own method falls short: its released update does not minimise its own objective. So no extra experiment was needed.

## Files

- **Gym:** `eval/train/evaluate.py`, `eval/test/evaluate.py` (independent copies; the test holds the hidden seeds), `method/embed.py`, `baseline/embed.py`, `data/directed_tree.py` (public family generator), `data/initialization.py`, `scripts/*.sh`, `scoring/reference.json`, `outputs/finsler_mds/method.json` (tracked reference card).
- **Removed:** `eval/tree_metrics.py`, `method/run_tree_experiment.py`, `baseline/run_tree_baseline.py` (artifact-file interface).
- **Score files:** `scoring/scores.json`, `scores_train.json` and `scores_test.json` come from the reference runs alone (`finsler_mds` plus the evaluator's `euclidean_mds`).
