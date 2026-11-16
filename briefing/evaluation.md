# Evaluation

## How Methods Are Evaluated

The evaluators **run your method** on the instances themselves; they never read results you computed in advance.

1. Register the method with a card anywhere under `outputs/`, e.g. `outputs/my_method/method.json`:

   ```json
   {"name": "my_method", "command": ["python", "-m", "my_package.embed", "--some-flag"]}
   ```

2. The evaluator appends `--instances <dir> --output <dir>` and runs the command from the workspace root (no network; stopped after 4 hours per slice as a hang guard). Temporary instance and output files go under `TMPDIR`.
3. `<instances>/index.json` lists `{"id", "file", "embedding_dim"}` entries; each instance file (`.npz`) holds `dissimilarities` (N x N directed distances, row = source) and an integer `seed`. `embedding_dim` is 2 for tree instances and 10 for citation instances.
4. For every instance the method writes `<output>/<id>.npz` with
   - `embedding`: finite array of shape (N, embedding_dim), and
   - `drift`: vector of length embedding_dim with norm < 1 (or `alpha`: a scalar, meaning drift `(0, ..., 0, alpha)`).

   Everything else in the file is ignored. Predicted directed distances are `d(i -> j) = ||x_j - x_i|| + drift . (x_j - x_i)`.

`baseline/embed.py` is a complete example of the interface.

Scripts (no arguments):
- `bash scripts/evaluate.sh`: the paper setting (exact depth-7 and depth-5 trees, whole Cora and Citeseer components), written to `scoring/scores.json`.
- `bash scripts/evaluate_train.sh`: the visible slice (8 depth-7 and 12 depth-5 trees, 5 Cora and 5 Citeseer subgraphs), written to `scoring/scores_train.json`.
- `bash scripts/evaluate_test.sh`: hidden instances from the same families (12 + 20 trees, 10 + 10 subgraphs), written to `scoring/scores_test.json`.

Every score file also contains `euclidean_mds`, a symmetric Euclidean SMACOF baseline (fitted to `(D + D^T)/2` in the same dimension) that the evaluator runs from its own frozen copy.

## Metrics (per instance, averaged over the experiment's instances)

Shared:
- `normalized_stress` (lower is better): `sqrt(sum (d - D)^2 / sum D^2)` over all ordered node pairs. Global reconstruction error.
- `distortion` (lower is better): mean over ordered pairs of `(d - D)^2 / D`, the distortion measure of the paper's Table 1. It weights short distances, so it reflects local structure.

Trees:
- `symmetric_rmse` (lower is better): RMSE of `(d_ij + d_ji)/2` against `(D_ij + D_ji)/2`.
- `asymmetry_rmse` (lower is better): RMSE of `d_ij - d_ji` against `D_ij - D_ji`.
- `distance_preservation_pass` (higher is better): `clip(1 - normalized_stress / 0.10, 0, 1)`.
- `asymmetry_preservation_pass`, `hierarchy_direction_pass` (higher is better): fraction of instances meeting `asymmetry_rmse <= 0.10` and `hierarchy_alignment >= 0.80`.
- Diagnostics, reported but not weighted: `map_at_10` (tie-aware MAP of the 10 nearest directed neighbours), `asymmetry_correlation`, `hierarchy_alignment` (signed agreement, over pairs of nodes on different levels, between the hierarchy and each node's uphill potential; 0 for symmetric distances).

Citation graphs:
- `map` (higher is better): the paper's Table 1 MAP. For every paper that cites another paper of the instance, rank all other papers by predicted directed distance from it and take the average precision of the papers it cites; ties are broken pessimistically.
- `asymmetry_error` (lower is better): RMSE of `(d_ij - d_ji) - (D_ij - D_ji)` divided by the RMS of `D_ij - D_ji`; exactly 1.0 for any symmetric embedding.
- `euclidean_gain_pass` (higher is better): fraction of instances with both lower distortion and higher MAP than `euclidean_mds` on the same instance.
- Diagnostic, not weighted: `symmetric_error` (as `asymmetry_error`, for the symmetric part).

Weights and coefficients are in `scoring/reference.json`.

## Eligibility

A method is eligible only if, on **every** instance of **every** experiment, it returns a valid output (right shape, finite, drift norm < 1, command exited with status 0) and meets its experiment's gate:
- trees: `normalized_stress <= 0.10`, `distortion <= 0.10`, `asymmetry_rmse <= 0.10`, `hierarchy_alignment >= 0.80`;
- citation graphs: `normalized_stress <= 0.25`, `asymmetry_error <= 0.80`, and stress, distortion and asymmetry error better than the empty prediction.

A method that is not eligible is scored as failed: every metric of every experiment takes the value of the empty (all-zero) prediction, which is worse than any eligible value.

## Baselines and Targets

The symmetric Euclidean baseline cannot represent direction, so it is never eligible (tree stress about 0.41-0.44, citation asymmetry error 1.0); its raw metrics are in the `details*.json` files. Reference (Finsler SMACOF) on the hidden slice: tree stress 0.023 (depth 7) and 0.027 (depth 5); Cora distortion 0.135, MAP 0.440, stress 0.109, asymmetry error 0.40; Citeseer 0.129, 0.492, 0.086, 0.41. Euclidean on the same subgraphs: Cora distortion 0.235, MAP 0.426; Citeseer 0.222, 0.443.

The paper reports the tree experiment qualitatively, so `scoring/reference.json` encodes those claims as pass checks. For the citation experiments it lists the paper's Table 1 values at dimension 10. They come from a neural embedding of the whole graph with the paper's own distance handling, so they are context, not targets on this gym's scale.
