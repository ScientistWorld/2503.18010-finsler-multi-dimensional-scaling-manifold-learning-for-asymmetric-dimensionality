# Progress

## What Works

- Paper analysis is complete for the minimum SMACOF reproduction path.
- The official code release is vendored under `method/upstream_finsler_mds`.
- Local tracked code now implements the Appendix D.1.1 directed binary-tree benchmark, the paper's canonical Randers Finsler SMACOF update, a Euclidean SMACOF baseline, and an evaluator that reads saved artifacts.
- CPU job `215fb11e-2bf` ran `scripts/reproduce.sh` successfully inside the container and `python validate.py --compare` passed.
- CPU job `fd4957cf-552` successfully regenerated `scoring/scores.json` with both the original depth-7 tree benchmark and an additional depth-5 tree benchmark.

## Results

- `directed_tree_reconstruction` (depth 7):
  - Euclidean SMACOF baseline: normalized stress `0.5028`, MAP@10 `0.2669`, asymmetry RMSE `1.8698`, symmetric RMSE `0.5379`, hierarchy alignment `0.0082`.
  - Finsler MDS: normalized stress `0.0222`, MAP@10 `0.2679`, asymmetry RMSE `0.0188`, symmetric RMSE `0.0466`, hierarchy alignment `0.9249`.
- `directed_tree_depth5_reconstruction`:
  - Euclidean SMACOF baseline: normalized stress `0.4949`, MAP@10 `0.6148`, asymmetry RMSE `1.6849`, symmetric RMSE `0.4496`, hierarchy alignment `0.0231`.
  - Finsler MDS: normalized stress `0.0254`, MAP@10 `0.5894`, asymmetry RMSE `0.0197`, symmetric RMSE `0.0481`, hierarchy alignment `0.9222`.
- These results support the paper's central claim that a canonical Randers Finsler geometry can represent directed asymmetric distances that a symmetric Euclidean embedding cannot.

## Remaining

- Full reproduction of Table 1 graph datasets remains future work; the packaged environment currently targets the paper's Appendix D.1.1 hierarchy benchmark plus one scaled setting.
- To reach `majority` or higher, implement the paper's directed-graph suite for Cora/Citeseer/Pubmed/Airports/Communities and add the neural embedding variants reported in Table 1.

## Issues

- The login-node Python environment does not have NumPy installed, so numerical experiments are run through scheduled jobs. Dependencies are declared in `environment/setup.sh` for the scheduled container job.
- The tree visualization experiment in the paper is reported qualitatively rather than with a numeric table. `reference.json` includes paper Table 1 Cora numbers and a benchmark target entry for the packaged tree task.
- CPU job `aa29f577-e55` failed before scoring because `scripts/evaluate.sh` still passed removed `--finsler` and `--euclidean` options. The script now calls the evaluator's default experiment suite.

## Deviations from Paper

<!-- Do not edit this section unless you deviated from the paper's original setup.
     For each deviation, document: what the paper used, what you substituted, and why.

     ACCEPTABLE deviations (data/resource/scale substitutions — milestones still valid):
       - Replaced a proprietary dataset with a public one
       - Used a smaller model of the same architecture
       - Scaled down training (fewer epochs, smaller batch)
       - Used a different library for the same algorithm

     UNACCEPTABLE deviations (algorithm substitutions — milestones above 'none' are void):
       - Replaced the paper's algorithm with a fundamentally different one
       - Wrote a generic model trained to approximate the method's outputs
         instead of implementing the method's computational steps

     The test: could someone reading your code identify which algorithm from
     the paper you implemented? If not, you replaced the method. -->

- The first executable reproduction uses the paper's directed binary-tree visualization task rather than the full neural directed-graph experiments. This preserves the paper's Finsler SMACOF algorithm and Appendix D.1.1 procedure while avoiding large graph representation-learning runs before the core method is validated.
- The evaluator adds explicit numeric metrics for the qualitative Figure 5 tree result: normalized stress, MAP@10, asymmetry RMSE, symmetric RMSE, and hierarchy alignment. These quantify the same reconstruction and hierarchy-preservation claim shown visually in the paper.
