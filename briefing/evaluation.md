# Evaluation

## How Methods Are Evaluated

The evaluators **run your method** on the instances themselves; they never read results you computed in advance.

1. Register the method with a card anywhere under `outputs/`, e.g. `outputs/my_method/method.json`:

   ```json
   {"name": "my_method", "command": ["python", "-m", "my_package.embed", "--some-flag"]}
   ```

2. The evaluator appends `--instances <dir> --output <dir>` and runs the command from the workspace root (no network, one CPU, at most 3600 s for all instances of a slice).
3. `<instances>/index.json` lists `{"id", "file"}` entries; each instance file (`.npz`) holds `dissimilarities` (N x N directed distances, row = source) and an integer `seed`.
4. For every instance the method writes `<output>/<id>.npz` with
   - `embedding`: finite array of shape (N, 2), and
   - `drift`: 2-vector with norm < 1 (or `alpha`: a scalar, meaning drift `(0, alpha)`).

   Everything else in the file is ignored. Predicted directed distances are `d(i -> j) = ||x_j - x_i|| + drift . (x_j - x_i)`.

`baseline/embed.py` is a complete example of the interface.

Scripts (no arguments):
- `bash scripts/evaluate.sh`: the paper's exact trees (depth 7 and depth 5), written to `scoring/scores.json`.
- `bash scripts/evaluate_train.sh`: the visible instance family (8 depth-7 and 12 depth-5 trees), written to `scoring/scores_train.json`.
- `bash scripts/evaluate_test.sh`: hidden instances from the same family (more of them), written to `scoring/scores_test.json`.

Every score file also contains `euclidean_mds`, a symmetric Euclidean SMACOF baseline that the evaluator runs from its own frozen copy.

## Metrics (per instance, averaged over the experiment's instances)

- `normalized_stress` (lower is better): `sqrt(sum (d - D)^2 / sum D^2)` over all ordered node pairs. Global reconstruction error.
- `distortion` (lower is better): mean over ordered pairs of `(d - D)^2 / D`, the distortion measure of the paper's digraph experiments. It weights short distances, so it reflects local structure.
- `symmetric_rmse` (lower is better): RMSE of `(d_ij + d_ji)/2` against `(D_ij + D_ji)/2`.
- `asymmetry_rmse` (lower is better): RMSE of `d_ij - d_ji` against `D_ij - D_ji`.
- `distance_preservation_pass` (higher is better): `clip(1 - normalized_stress / 0.10, 0, 1)`.
- `asymmetry_preservation_pass`, `hierarchy_direction_pass` (higher is better): fraction of instances meeting `asymmetry_rmse <= 0.10` and `hierarchy_alignment >= 0.80`.
- Diagnostics, reported but not weighted:
  - `map_at_10`: tie-aware MAP of the 10 nearest directed neighbours.
  - `asymmetry_correlation`.
  - `hierarchy_alignment`: signed agreement, over pairs of nodes on different levels, between the hierarchy and each node's uphill potential (total incoming minus total outgoing predicted distance). It is 0 for symmetric distances.

## Eligibility

A method is eligible only if, on **every** instance of **both** experiments, it returns a valid output (right shape, finite, drift norm < 1, command exited with status 0) and meets all of: `normalized_stress <= 0.10`, `distortion <= 0.10`, `asymmetry_rmse <= 0.10`, `hierarchy_alignment >= 0.80`. A method that is not eligible is scored as failed: every metric of every experiment takes the value of the empty (all-zero) prediction, which is worse than any eligible value.

## Baselines and Targets

The symmetric Euclidean baseline cannot represent direction. Its raw stress is about 0.48 and its asymmetry RMSE about 1.5-1.9, so it is not eligible. The paper reports the tree experiment qualitatively. `scoring/reference.json` therefore encodes those qualitative claims as pass checks and keeps the paper's quantitative digraph numbers (other datasets) under `_paper_table1_2d_reconstruction` for context only.
