# Evaluation Targets

## Primary: directed_tree_reconstruction
Tests whether a low-dimensional embedding reconstructs directed shortest-path dissimilarities on the hierarchy benchmark.
- **Metric**: `distance_preservation_pass` (higher is better)
- **Run**: `bash scripts/reproduce.sh`, or `bash scripts/evaluate.sh` after method artifacts exist.

## Primary: directed_tree_depth5_reconstruction
Repeats the hierarchy reconstruction target on a smaller directed tree to check that the effect is not limited to one graph size.
- **Metric**: `distance_preservation_pass` (higher is better)
- **Run**: `bash scripts/reproduce.sh`, or `bash scripts/evaluate.sh` after method artifacts exist.

## Constraint: directed_tree_reconstruction
Tests whether the embedding preserves directional asymmetry rather than fitting only the symmetric part of the distances.
- **Metric**: `asymmetry_preservation_pass` (higher is better)
- **Run**: `bash scripts/evaluate.sh`

## Constraint: directed_tree_depth5_reconstruction
Tests directional-asymmetry preservation on the smaller directed-tree instance.
- **Metric**: `asymmetry_preservation_pass` (higher is better)
- **Run**: `bash scripts/evaluate.sh`

## Secondary: directed_tree_reconstruction
Tests whether the embedding preserves local directed neighbour rankings and exposes the hierarchy ordering.
- **Metric**: `hierarchy_direction_pass` (higher is better)
- **Run**: `bash scripts/evaluate.sh`

## Secondary: directed_tree_depth5_reconstruction
Tests local directed-neighbour rankings and hierarchy ordering on the smaller tree.
- **Metric**: `hierarchy_direction_pass` (higher is better)
- **Run**: `bash scripts/evaluate.sh`

Continuous diagnostics used to compute these pass/fail checks are written to `scoring/details.json`.
