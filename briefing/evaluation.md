# Evaluation

## Metrics

- `normalized_stress`: square-root stress normalized by the squared target dissimilarities. Lower is better.
- `map_at_10`: mean average precision of the top-10 nearest directed neighbours under the predicted distances. Higher is better.
- `asymmetry_rmse`: RMSE between target and predicted antisymmetric distance components, `D_ij - D_ji`. Lower is better.
- `symmetric_rmse`: RMSE between target and predicted symmetric distance components. Lower is better.
- `hierarchy_alignment`: absolute Spearman correlation between the embedding's final coordinate and tree depth on the hierarchy benchmark. Higher is better.
- `distance_preservation_pass`: binary Figure 5 claim check derived from normalized stress. Higher is better.
- `asymmetry_preservation_pass`: binary Figure 5 claim check derived from asymmetry RMSE. Higher is better.
- `hierarchy_direction_pass`: binary Figure 5 claim check derived from hierarchy alignment. Higher is better.

## Baselines and Targets

The primary baseline is a Euclidean low-dimensional embedding optimized against the same directed dissimilarity matrix. The paper's Table 1 reports that on Cora with 2D embeddings, Euclidean reconstruction has distortion `0.426` and MAP `0.574`, while the proposed directed-distance embedding has distortion `0.127` and MAP `0.894`.

## Evaluation Protocol

For the packaged reproduction, run the fixed directed binary-tree benchmark from the paper's visualization experiments. Generate the graph, compute all-pairs directed shortest-path dissimilarities, run the baseline and proposed method with the same seed and embedding dimension, then evaluate only the saved output artifacts with `scripts/evaluate.sh`.

The paper reports the tree result qualitatively rather than as a numeric table. To avoid inventing continuous paper values, `scoring/scores.json` reports binary checks for the qualitative Figure 5 claims, while `scoring/details.json` records the continuous normalized stress, MAP@10, asymmetry RMSE, symmetric RMSE, and hierarchy alignment diagnostics.
