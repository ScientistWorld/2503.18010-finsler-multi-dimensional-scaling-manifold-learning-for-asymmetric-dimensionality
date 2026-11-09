This audit uses improve mode. Candidate implementations are in `audit.py` and
each named folder's `run.sh` installs that candidate, runs the public train
script, then invokes the unchanged `bash scripts/evaluate_test.sh` with **no
arguments**, in a fresh process. No candidate reads hidden score values or
hidden split code. `summarize.py` is audit-only analysis after all runs.

Run all experiments on a compute node with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m redteam.audit all
python -m redteam.summarize
```

The local ignored `.state/` backs up the starting artifacts and scoring files;
`.cache/` retains generated artifacts for optional replay. Each experiment
retains its actual train/test score JSON and exit status. A standalone case
can be rerun using `bash redteam/<case>/run.sh`. Original outputs must exist;
if they do not, regenerate the baseline and reference using the original
`scripts/run.sh` first. Failed or incomplete test submissions receive no
invented score. The fixed-generator shortcut also records its test scores
against the original task after restoring the generator.

All shortcuts are deliberately named `shortcut_*`, including exact ties.
The audit never edits the evaluator, its process, or the Python installation.
The only problem-file mutation is the explicitly tested visible data asset.
Job control temporarily replaces `scripts/run.sh` and `action.yaml` as directed
by the audit task. They are restored after the experiment job (with the final
required `action: done` signal).
