# Evaluation Targets

## Primary: directed_tree_reconstruction
Tests whether a low-dimensional embedding reconstructs directed shortest-path dissimilarities on the hierarchy benchmark.
- **Metric**: `normalized_stress` (lower is better)
- **Run**: `bash scripts/reproduce.sh`, or `bash scripts/evaluate.sh` after method artifacts exist.

## Constraint: directed_tree_reconstruction
Tests whether the embedding preserves directional asymmetry rather than fitting only the symmetric part of the distances.
- **Metric**: `asymmetry_rmse` (lower is better)
- **Run**: `bash scripts/evaluate.sh`

## Secondary: directed_tree_reconstruction
Tests whether the embedding preserves local directed neighbour rankings and exposes the hierarchy ordering.
- **Metrics**: `map_at_10` and `hierarchy_alignment` (higher is better)
- **Run**: `bash scripts/evaluate.sh`
