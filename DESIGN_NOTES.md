> **Superseded (2026-11-09):** the evaluation design described below (artifact files, source-row split of the public trees) was replaced by the hardened design in `HARDENING.md` and `briefing/evaluation.md` (methods are run on hidden instances; scores are recomputed from 2D embeddings with an eligibility gate).

# Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side.
- Confirmed `evaluate_test.sh` invokes `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Confirmed `evaluate_train.sh` invokes `eval.train.evaluate` and writes `scoring/scores_train.json`.
- Checked there is no hardcoded method default; both slices discover `outputs/**/predictions.npz` or accept a manifest.
- Checked the evaluators score saved artifacts only and do not import or rerun `method/`.
- Checked whether evaluators trusted method-supplied target matrices; they did before this pass.
- Grepped for train/test cross-imports; only the shell entry points import their own slice modules.
- Grepped `scoring/reference.json` and eval code for paper-agreement metric names; none were present.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n` on both train/test shell scripts.

## Changes Made

- Added continuous official score metrics to both train and test evaluators:
  `normalized_stress`, `map_at_10`, `asymmetry_rmse`, `asymmetry_correlation`,
  `symmetric_rmse`, and `hierarchy_alignment`.
- Kept the original binary qualitative claim checks, but they are no longer the only scored signal.
- Added `asymmetry_correlation` as a second directional-structure metric, so a method has to preserve both the magnitude and ordering of directed components.
- Stopped trusting `distances` and `depths` inside method artifacts. The evaluators now derive the fixed directed-tree target and split depths from neutral `data.directed_tree` code using the artifact's node count, and read only predictions or embeddings from the method artifact.
- Updated `scoring/reference.json` with descriptions for every metric.
- Reweighted metric coefficients so continuous scientific quantities dominate the score and binary pass/fail checks act as guards rather than the whole target.
- Removed the Cora reference experiment from the train/test split schema because the current split only enumerates directed-tree artifacts; leaving it in `reference.json` created an uncomputed, gameable target.
- Rebalanced implemented experiment weights to `0.6` for the depth-7 tree and `0.4` for the depth-5 tree.

## Runtime Concerns For Next Iteration

- I did not run the evaluators or generate score files, per the design-only instruction.
- The split is a stratified source-row split over the same public procedural tree instances. A method that only fills the visible train source rows should regress on the held-out complement, but a method that hardcodes the full directed-tree distance formula for the two public node counts can still pass both slices. A stronger future design would require precomputed neutral evaluation assets or a method interface that can emit predictions for hidden procedural instances without retraining.
- `hierarchy_alignment` depends on an `embedding`; artifacts with only `predicted_distances` receive `0.0` for that metric. That is intentional for methods that cannot expose a hierarchy coordinate, but runner results should confirm it does not unfairly suppress legitimate non-coordinate improvements.
- The train/test score schema now contains continuous metrics that are not present in the paper result records in `reference.json`; this avoids modifying paper-reported results, but the next runner should confirm the benchmark scorer handles missing reference values for non-paper metrics as intended.
