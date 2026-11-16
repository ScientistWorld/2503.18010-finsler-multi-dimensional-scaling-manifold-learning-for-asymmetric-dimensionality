# Evaluation Targets

Experiment weights: `directed_tree_reconstruction` 0.30, `directed_tree_depth5_reconstruction` 0.20, `cora_digraph_reconstruction` 0.25, `citeseer_digraph_reconstruction` 0.25. Run everything with `bash scripts/reproduce.sh`, or `bash scripts/evaluate.sh` / `evaluate_train.sh` / `evaluate_test.sh` once a method card exists.

## Primary: directed_tree_reconstruction
Reconstruct directed shortest-path distances on depth-7 hierarchies from a 2D representation.
- **Metric**: `distance_preservation_pass` (higher is better; graded clearance of `normalized_stress` below 0.10). The underlying continuous metrics `normalized_stress` and `distortion` (lower is better) carry most of the weight.

## Primary: directed_tree_depth5_reconstruction
The same target on depth-5 hierarchies.
- **Metric**: `distance_preservation_pass` (higher is better), with `normalized_stress` and `distortion` (lower is better).

## Primary: cora_digraph_reconstruction
Reconstruct directed citation-graph distances from a 10-dimensional representation (paper Table 1).
- **Metric**: `distortion` (lower is better; coefficient -1.0). Benefit metric `map` (higher is better; +1.0) and `normalized_stress` (lower is better; -0.5).

## Primary: citeseer_digraph_reconstruction
- **Metric**: `distortion` (lower is better), with `map` (higher is better) and `normalized_stress` (lower is better), as for Cora.

## Constraint: asymmetry (all experiments)
The directional (asymmetric) part of the distances must be preserved, not only the symmetric part.
- **Trees**: `asymmetry_preservation_pass` (higher is better), from `asymmetry_rmse <= 0.10` on every instance; `asymmetry_rmse` (lower is better) is also weighted.
- **Citation graphs**: `asymmetry_error` (lower is better; -0.3), the antisymmetric reconstruction error relative to the target's antisymmetric part. It is exactly 1.0 for any symmetric representation and must stay `<= 0.80` on every instance.

## Constraint: hierarchy direction (tree experiments)
The hierarchy and its direction must be readable from the predicted distances.
- **Metric**: `hierarchy_direction_pass` (higher is better), from `hierarchy_alignment >= 0.80` on every instance.

## Constraint: baseline comparison (citation experiments)
- **Metric**: `euclidean_gain_pass` (higher is better; +0.3): fraction of instances on which the method has both lower distortion and higher MAP than the evaluator's frozen Euclidean embedding of the same instance.

## Secondary and diagnostics
- `symmetric_rmse` (trees, weighted) and `symmetric_error` (citation graphs, zero weight).
- `map_at_10`, `asymmetry_correlation` and `hierarchy_alignment` (trees) are reported with zero weight.

## Eligibility
A method is ranked only if it returns a valid output for every instance of every experiment and meets every gate on every instance. Trees: `normalized_stress <= 0.10`, `distortion <= 0.10`, `asymmetry_rmse <= 0.10`, `hierarchy_alignment >= 0.80`. Citation graphs: `normalized_stress <= 0.25`, `asymmetry_error <= 0.80`, and stress, distortion and asymmetry error better than the empty prediction. Otherwise every metric of every experiment takes the value of the empty (all-zero) prediction.

## Artifact Interface
Methods are run by the evaluators. Register a card `outputs/<name>/method.json` = `{"name": "<name>", "command": [...]}`. The evaluator appends `--instances <dir> --output <dir>`; `index.json` gives each instance's `embedding_dim` (2 for trees, 10 for citation graphs). The method writes `<output>/<id>.npz` with `embedding` (N x embedding_dim) and `drift` (embedding_dim-vector, norm < 1) or `alpha` (scalar, drift along the last axis). See `briefing/evaluation.md` and `baseline/embed.py`.
