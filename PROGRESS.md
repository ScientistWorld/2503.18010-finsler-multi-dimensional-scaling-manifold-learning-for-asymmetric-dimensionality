# Progress

## What Works

- Paper analysis is complete for the minimum SMACOF reproduction path.
- The official code release is vendored under `method/upstream_finsler_mds`.
- Local tracked code now implements the Appendix D.1.1 directed binary-tree benchmark, the paper's canonical Randers Finsler SMACOF update, a Euclidean SMACOF baseline, and an evaluator that reads saved artifacts.
- `python validate.py --reference-only` passes before the first job submission.

## Results

- No reproduced numbers yet. The first CPU test job will run `scripts/reproduce.sh` and write `scoring/scores.json`.

## Remaining

- Run the CPU test job and inspect `scoring/scores.json`.
- If the Finsler method improves directed-tree reconstruction and asymmetry preservation over the Euclidean baseline, update the milestone to `core_claim`.
- If the first run is unstable, tune initialization or iterations while keeping the same Finsler SMACOF procedure.

## Issues

- The login-node Python environment does not have NumPy installed, so no local numerical smoke test was run. Dependencies are declared in `environment/setup.sh` for the scheduled container job.
- The tree visualization experiment in the paper is reported qualitatively rather than with a numeric table. `reference.json` includes paper Table 1 Cora numbers and a benchmark target entry for the packaged tree task.

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
