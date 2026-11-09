# Hardening audit

Replays genuine variants and shortcuts against the hardened gym. Every candidate
is installed only as a method card under `outputs/audit/<name>/method.json` and
scored by the unchanged `bash scripts/evaluate_train.sh` and
`bash scripts/evaluate_test.sh` (no arguments). Three shortcuts that modify
problem files run in isolation and are restored afterwards:
- `fixed_generator`: patches `data/directed_tree.py`;
- `move_baseline`: patches `baseline/euclidean_smacof.py`;
- `train_filter`: empties `scoring/scores_train.json`.

```bash
bash scripts/method.sh                         # reference card must exist
python -m redteam.hardening.audit run          # batch + isolated + summary (~45 min, 1 CPU)
python -m redteam.hardening.audit summarize    # pure-Python rank analysis only
```

- `variants.py`: corrected-majorization SMACOF, L-BFGS and multi-scale refinements, level-collapse probe.
- `shortcuts.py`: the shortcut submissions.
- `results/`:
  - `official/`: the reference-only score files from the first pass;
  - `batch/`: every card, train and test;
  - `fixed_generator/`, `move_baseline/`, `train_filter/`: the isolated cases;
  - `summary.json`, `results_table.md`: ranks in the eligible, genuine and all-attempt pools, seed spread, paired gains.

The audit overwrites `scoring/scores_*.json`; rerun `bash scripts/reproduce.sh`
afterwards to restore the reference-only score files. See `HARDENING.md`.
