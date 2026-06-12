# Evaluation

## Metrics

- `normalized_stress`: square-root stress normalized by the squared target dissimilarities. Lower is better.
- `map_at_10`: mean average precision of the top-10 nearest directed neighbours under the predicted distances. Higher is better.
- `asymmetry_rmse`: RMSE between target and predicted antisymmetric distance components, `D_ij - D_ji`. Lower is better.
- `symmetric_rmse`: RMSE between target and predicted symmetric distance components. Lower is better.
- `hierarchy_alignment`: absolute Spearman correlation between the embedding's final coordinate and tree depth on the hierarchy benchmark. Higher is better.
- `distance_preservation_pass`: threshold-clearance score derived from normalized stress; zero means the reconstruction error did not clear the Figure 5 claim threshold. Higher is better.
- `asymmetry_preservation_pass`: binary Figure 5 claim check derived from asymmetry RMSE. Higher is better.
- `hierarchy_direction_pass`: binary Figure 5 claim check derived from hierarchy alignment. Higher is better.

## Baselines and Targets

The reference baseline is a symmetric Euclidean low-dimensional embedding optimized against the same directed dissimilarity matrix. The paper's larger graph experiments report that directed-distance-preserving embeddings can substantially reduce distortion and improve directed-neighbour ranking compared with symmetric embeddings.

## Evaluation Protocol

For the packaged reproduction, run the fixed directed binary-tree benchmark from the visualization experiments. Generate the graph, compute all-pairs directed shortest-path dissimilarities, run each candidate embedding method with the same seed and embedding dimension, then evaluate only the saved output artifacts with `scripts/evaluate.sh`.

The paper reports the tree result qualitatively rather than as a numeric table. To avoid inventing continuous paper values, `scoring/reference.json` records the qualitative Figure 5 expectations as threshold checks. `scoring/scores.json` records artifact-derived continuous diagnostics plus those claim checks, and `scoring/details.json` keeps the same diagnostics with threshold metadata.
