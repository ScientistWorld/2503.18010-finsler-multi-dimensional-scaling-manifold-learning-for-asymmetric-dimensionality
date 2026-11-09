# Evaluation Targets

## Primary: directed_tree_reconstruction
Reconstruct directed shortest-path distances on depth-7 hierarchies from a 2D representation.
- **Metric**: `distance_preservation_pass` (higher is better; graded clearance of `normalized_stress` below 0.10). The underlying continuous metrics `normalized_stress` and `distortion` (lower is better) carry most of the weight.
- **Run**: `bash scripts/reproduce.sh`, or `bash scripts/evaluate.sh` / `evaluate_train.sh` / `evaluate_test.sh` once the method card exists.

## Primary: directed_tree_depth5_reconstruction
The same target on depth-5 hierarchies.
- **Metric**: `distance_preservation_pass` (higher is better), with `normalized_stress` and `distortion` (lower is better).
- **Run**: as above.

## Constraint: directed_tree_reconstruction
The directional (asymmetric) part of the distances must be preserved, not only the symmetric part.
- **Metric**: `asymmetry_preservation_pass` (higher is better), from `asymmetry_rmse <= 0.10` on every instance; `asymmetry_rmse` (lower is better) is also weighted.
- **Run**: `bash scripts/evaluate_test.sh`

## Constraint: directed_tree_depth5_reconstruction
- **Metric**: `asymmetry_preservation_pass` (higher is better) and `asymmetry_rmse` (lower is better).
- **Run**: `bash scripts/evaluate_test.sh`

## Constraint: hierarchy direction (both experiments)
The hierarchy and its direction must be readable from the predicted distances.
- **Metric**: `hierarchy_direction_pass` (higher is better), from `hierarchy_alignment >= 0.80` on every instance.
- **Run**: `bash scripts/evaluate_test.sh`

## Secondary: symmetric structure (both experiments)
- **Metric**: `symmetric_rmse` (lower is better).

## Eligibility
A method that returns an invalid output for any instance or misses any of `normalized_stress <= 0.10`, `distortion <= 0.10`, `asymmetry_rmse <= 0.10`, `hierarchy_alignment >= 0.80` on any instance is scored as failed on every metric. `map_at_10`, `asymmetry_correlation` and `hierarchy_alignment` are reported as diagnostics with zero weight.

## Artifact Interface
Methods are run by the evaluators. Register a card `outputs/<name>/method.json` = `{"name": "<name>", "command": [...]}`. The evaluator appends `--instances <dir> --output <dir>`; the method writes `<output>/<id>.npz` with `embedding` (N x 2) and `drift` (2-vector, norm < 1) or `alpha` (scalar). See `briefing/evaluation.md` and `baseline/embed.py`.
