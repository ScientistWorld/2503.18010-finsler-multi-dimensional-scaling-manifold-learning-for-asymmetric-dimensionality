#!/usr/bin/env python
"""Hardening audit: replay genuine variants and shortcuts through the hardened gym.

Every candidate is installed only as a method card under ``outputs/audit/`` and
scored by the unchanged ``bash scripts/evaluate_train.sh`` /
``bash scripts/evaluate_test.sh`` (no arguments). Three shortcuts that modify
problem files (the public generator, the public baseline, the train score
file) run in isolation and are restored afterwards.

    python -m redteam.hardening.audit run        # batch + isolated cases
    python -m redteam.hardening.audit summarize  # rank analysis (pure Python)
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "redteam" / "hardening"
RESULTS = HERE / "results"
AUDIT_CARDS = ROOT / "outputs" / "audit"
REFERENCE = "finsler_mds"
BASELINE = "euclidean_mds"

REF = ["python", "-m", "method.embed", "--alpha", "0.5", "--max-iter", "500", "--eps", "1e-5"]
VAR = ["python", "-m", "redteam.hardening.variants", "--variant"]
SHORT = ["python", "-m", "redteam.hardening.shortcuts", "--kind"]

GENUINE = {
    **{f"ref_seed{k}": REF + ["--seed-offset", str(k)] for k in range(1, 5)},
    "better_exact_smacof": VAR + ["exact_smacof"],
    "better_exact_smacof_seed1": VAR + ["exact_smacof", "--seed-offset", "1"],
    "better_exact_smacof_seed2": VAR + ["exact_smacof", "--seed-offset", "2"],
    "better_lbfgs": VAR + ["lbfgs"],
    "better_lbfgs_seed1": VAR + ["lbfgs", "--seed-offset", "1"],
    "better_lbfgs_seed2": VAR + ["lbfgs", "--seed-offset", "2"],
    "better_multiscale": VAR + ["multiscale"],
    "better_multiscale_seed1": VAR + ["multiscale", "--seed-offset", "1"],
    "better_multiscale_seed2": VAR + ["multiscale", "--seed-offset", "2"],
    "better_long_smacof": ["python", "-m", "method.embed", "--alpha", "0.5", "--max-iter", "2000", "--eps", "1e-10"],
    "weaker_10iter": ["python", "-m", "method.embed", "--alpha", "0.5", "--max-iter", "10"],
    "weaker_5iter": ["python", "-m", "method.embed", "--alpha", "0.5", "--max-iter", "5"],
    "weaker_alpha03": ["python", "-m", "method.embed", "--alpha", "0.3", "--max-iter", "500"],
    "weaker_euclidean": ["python", "-m", "baseline.embed"],
    "probe_level_collapse": VAR + ["level_collapse"],
}
BATCH_SHORTCUTS = [
    "answer_copy", "hierarchy_field", "task_failure", "score_fields", "forged_targets",
    "impossible_diagonal", "nan", "inf", "empty", "wrong_dim", "drift_too_large",
    "missing_instance", "crash",
]
SEED_GROUPS = {
    "reference": [REFERENCE] + [f"ref_seed{k}" for k in range(1, 5)],
    "better_exact_smacof": ["better_exact_smacof", "better_exact_smacof_seed1", "better_exact_smacof_seed2"],
    "better_lbfgs": ["better_lbfgs", "better_lbfgs_seed1", "better_lbfgs_seed2"],
    "better_multiscale": ["better_multiscale", "better_multiscale_seed1", "better_multiscale_seed2"],
}
PAIRED_REFERENCE = [REFERENCE, "ref_seed1", "ref_seed2"]

EASY_GENERATOR = '''

# AUDIT PATCH: replace the public generator with an easy distance geometry.
def make_directed_binary_tree(depth=7, parent_to_child=0.5, child_to_parent=1.5, same_depth=0.1):
    n = 2 ** (depth + 1) - 1
    levels = np.floor(np.log2(np.arange(n) + 1)).astype(np.int64)
    x = np.column_stack([np.arange(n) * 1e-4, -levels.astype(float)])
    delta = x[None, :, :] - x[:, None, :]
    distances = np.linalg.norm(delta, axis=-1) + 0.5 * delta[..., -1]
    return DirectedTreeData(adjacency=distances.copy(), distances=distances, depths=levels)
'''
ZERO_BASELINE = '''

# AUDIT PATCH: sabotage the public baseline.
_original_smacof = smacof
def smacof(dissimilarities, n_components=2, **kwargs):
    n = dissimilarities.shape[0]
    return EuclideanResult(embedding=np.zeros((n, n_components)), stress_history=[0.0], n_iter=1)
'''


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _card(name: str, command: list[str] | None, raw: str | None = None) -> None:
    folder = AUDIT_CARDS / name
    folder.mkdir(parents=True, exist_ok=True)
    text = raw if raw is not None else json.dumps({"name": name, "command": command}, indent=1)
    (folder / "method.json").write_text(text)


def _clear_cards() -> None:
    if AUDIT_CARDS.exists():
        shutil.rmtree(AUDIT_CARDS)


def _run(script: str, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(["bash", script], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return proc.returncode


def _collect(dest: Path, names: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = ROOT / "scoring" / name
        if src.exists():
            shutil.copy2(src, dest / name)


def run_batch() -> None:
    _clear_cards()
    for name, command in GENUINE.items():
        _card(name, command)
    for kind in BATCH_SHORTCUTS:
        _card(f"shortcut_{kind}", SHORT + [kind])
    _card("shortcut_bad_card", None, raw='{"name": "shortcut_bad_card", "command": "python -c 1"')
    dest = RESULTS / "batch"
    status = {"train": _run("scripts/evaluate_train.sh", dest / "train.log")}
    _collect(dest, ["scores_train.json", "details_train.json"])
    # Forged score file: the evaluator must overwrite it from recomputed metrics.
    forged = json.loads((ROOT / "scoring" / "scores_train.json").read_text())
    for experiment in forged.values():
        for metrics in experiment["results"].values():
            for key in metrics:
                metrics[key] = 0.0 if key in {"normalized_stress", "distortion", "asymmetry_rmse", "symmetric_rmse"} else 1.0
    _dump(ROOT / "scoring" / "scores_test.json", forged)
    status["test"] = _run("scripts/evaluate_test.sh", dest / "test.log")
    _collect(dest, ["scores_test.json", "details_test.json"])
    _dump(dest / "exit_status.json", status)
    _clear_cards()


def run_isolated() -> None:
    # 1. Replace the public generator; the hidden test must not move.
    gen = ROOT / "data" / "directed_tree.py"
    original = gen.read_text()
    try:
        gen.write_text(original + EASY_GENERATOR)
        _card("shortcut_fixed_generator", SHORT + ["generator_easy"])
        dest = RESULTS / "fixed_generator"
        _dump(dest / "exit_status.json", {"test": _run("scripts/evaluate_test.sh", dest / "test.log")})
        _collect(dest, ["scores_test.json", "details_test.json"])
    finally:
        gen.write_text(original)
        _clear_cards()

    # 2. Sabotage the public baseline; the evaluator's frozen baseline must not move.
    base = ROOT / "baseline" / "euclidean_smacof.py"
    original = base.read_text()
    try:
        base.write_text(original + ZERO_BASELINE)
        _card("shortcut_move_baseline_public", ["python", "-m", "baseline.embed"])
        dest = RESULTS / "move_baseline"
        _dump(dest / "exit_status.json", {"test": _run("scripts/evaluate_test.sh", dest / "test.log")})
        _collect(dest, ["scores_test.json", "details_test.json"])
    finally:
        base.write_text(original)
        _clear_cards()

    # 3. Empty the method lists of the train score file; the test must still
    #    score every method on disk.
    train_scores = ROOT / "scoring" / "scores_train.json"
    saved = train_scores.read_text() if train_scores.exists() else None
    try:
        emptied = {exp: {"results": {}} for exp in json.loads(saved or "{}")}
        _dump(train_scores, emptied)
        _card("probe_level_collapse", VAR + ["level_collapse"])
        dest = RESULTS / "train_filter"
        _dump(dest / "exit_status.json", {"test": _run("scripts/evaluate_test.sh", dest / "test.log")})
        _collect(dest, ["scores_test.json", "details_test.json"])
    finally:
        if saved is not None:
            train_scores.write_text(saved)
        _clear_cards()


# --------------------------------------------------------------------------
# Rank analysis (pure Python so it also runs on the login node)
# --------------------------------------------------------------------------


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _contract() -> dict:
    ref = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    return {
        exp: {
            "weight": spec["weight"],
            "metrics": {
                m: (abs(float(d["coefficient"])), bool(d["higher_is_better"]))
                for m, d in spec["metrics"].items()
                if float(d["coefficient"]) != 0.0
            },
        }
        for exp, spec in ref.items()
    }


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def rank(scores: dict, contract: dict, pool: list[str], method: str, reference: dict) -> float:
    total = 0.0
    for exp, spec in contract.items():
        results = scores[exp]["results"]
        for metric, (coef, higher) in spec["metrics"].items():
            sd = _std([results[p][metric] for p in pool])
            if sd <= 0:
                continue
            diff = results[method][metric] - reference[exp][metric]
            total += spec["weight"] * coef * (diff if higher else -diff) / sd
    return total


def summarize() -> None:
    contract = _contract()
    test = _load(RESULTS / "batch" / "scores_test.json")
    train = _load(RESULTS / "batch" / "scores_train.json")
    details = _load(RESULTS / "batch" / "details_test.json")
    if test is None:
        print("no batch results", file=sys.stderr)
        return
    reference = {exp: test[exp]["results"][REFERENCE] for exp in test}
    methods = sorted(test[next(iter(test))]["results"])
    genuine = [REFERENCE] + [m for m in GENUINE if m in methods]
    attempts = [m for m in methods if m != BASELINE]
    shortcut_names = [m for m in methods if m.startswith("shortcut_")]

    def ranks_for(method: str) -> dict:
        pool_g = genuine if method in genuine else genuine + [method]
        return {
            "rank_genuine_pool": rank(test, contract, pool_g, method, reference),
            "rank_all_pool": rank(test, contract, attempts if method in attempts else attempts + [method], method, reference),
        }

    table = {}
    for method in methods:
        info = details["methods"].get(method, {}) if details else {}
        row = {
            "eligible": info.get("eligible"),
            "first_failure": (info.get("failure_reasons") or [None])[0],
            **ranks_for(method),
            "test": {exp: test[exp]["results"][method] for exp in test},
            "test_raw_mean": {exp: (info.get("experiments", {}).get(exp, {}) or {}).get("raw_mean") for exp in test},
        }
        if train and method in train[next(iter(train))]["results"]:
            row["train"] = {exp: train[exp]["results"][method] for exp in train}
        table[method] = row

    seed = {}
    for group, members in SEED_GROUPS.items():
        present = [m for m in members if m in table]
        if not present:
            continue
        for pool in ("rank_genuine_pool", "rank_all_pool"):
            vals = [table[m][pool] for m in present]
            seed.setdefault(group, {})[pool] = {
                "values": vals, "mean": sum(vals) / len(vals), "range": max(vals) - min(vals), "std": _std(vals),
            }
        per_metric = {}
        for exp in test:
            for metric in test[exp]["results"][REFERENCE]:
                vals = [test[exp]["results"][m][metric] for m in present]
                per_metric[f"{exp}.{metric}"] = {"mean": sum(vals) / len(vals), "std": _std(vals), "range": max(vals) - min(vals)}
        seed[group]["per_metric"] = per_metric

    paired = {}
    for group in ("better_exact_smacof", "better_lbfgs", "better_multiscale"):
        members = SEED_GROUPS[group]
        if not all(m in table for m in members + PAIRED_REFERENCE):
            continue
        for pool in ("rank_genuine_pool", "rank_all_pool"):
            diffs = [table[b][pool] - table[r][pool] for b, r in zip(members, PAIRED_REFERENCE)]
            paired.setdefault(group, {})[pool] = {"paired_differences": diffs, "min": min(diffs), "mean": sum(diffs) / len(diffs)}

    isolated = {}
    for case in ("fixed_generator", "move_baseline", "train_filter"):
        scores = _load(RESULTS / case / "scores_test.json")
        status = _load(RESULTS / case / "exit_status.json")
        det = _load(RESULTS / case / "details_test.json")
        isolated[case] = {
            "exit_status": status,
            "methods": sorted(scores[next(iter(scores))]["results"]) if scores else None,
            "eligible": {m: v["eligible"] for m, v in det["methods"].items()} if det else None,
            "scores": scores,
        }
    if isolated.get("move_baseline", {}).get("scores"):
        mb = isolated["move_baseline"]["scores"]
        isolated["move_baseline"]["frozen_baseline_unchanged"] = all(
            mb[exp]["results"][BASELINE] == test[exp]["results"][BASELINE] for exp in test
        )
    if isolated.get("fixed_generator", {}).get("scores"):
        fg = isolated["fixed_generator"]["scores"]
        isolated["fixed_generator"]["reference_unchanged"] = all(
            fg[exp]["results"][REFERENCE] == test[exp]["results"][REFERENCE] for exp in test
        )

    forged_overwritten = bool(test) and all(
        test[exp]["results"][REFERENCE]["normalized_stress"] > 0 for exp in test
    )
    summary = {
        "contract": {exp: {"weight": c["weight"], "metrics": {m: list(v) for m, v in c["metrics"].items()}} for exp, c in contract.items()},
        "pools": {"genuine": genuine, "all_attempts": attempts},
        "methods": table,
        "seed_spread": seed,
        "paired_better_minus_reference": paired,
        "isolated": isolated,
        "forged_score_file_overwritten": forged_overwritten,
        "shortcuts": shortcut_names,
    }
    _dump(RESULTS / "summary.json", summary)

    lines = [
        "| Attempt | Eligible | Rank (genuine pool) | Rank (all pool) | Stress d7 / d5 | Distortion d7 / d5 | Asym RMSE d7 / d5 | Sym RMSE d7 / d5 |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    exps = list(test)
    for method in [REFERENCE] + [m for m in methods if m != REFERENCE]:
        row = table[method]
        def pair(metric):
            return " / ".join(f"{row['test'][e][metric]:.6f}" for e in exps)
        lines.append(
            f"| {method} | {row['eligible']} | {row['rank_genuine_pool']:+.4f} | {row['rank_all_pool']:+.4f} | "
            f"{pair('normalized_stress')} | {pair('distortion')} | {pair('asymmetry_rmse')} | {pair('symmetric_rmse')} |"
        )
    (RESULTS / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "batch", "isolated", "summarize"])
    args = parser.parse_args()
    if args.command in ("run", "batch"):
        run_batch()
    if args.command in ("run", "isolated"):
        run_isolated()
    if args.command in ("run", "summarize"):
        summarize()


if __name__ == "__main__":
    main()
