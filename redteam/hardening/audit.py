#!/usr/bin/env python
"""Hardening audit: replay genuine variants and shortcuts through the hardened gym.

Every candidate is installed only as a method card under ``outputs/audit/`` and
scored by the unchanged ``bash scripts/evaluate_train.sh`` /
``bash scripts/evaluate_test.sh`` (no arguments). Shortcuts that modify
problem files (the public tree generator, the public citation graphs, the
public baseline, the train score file) run in isolation and are restored
afterwards. The official score files are backed up first and restored at the
end, so the audit never changes them.

    python -m redteam.hardening.audit run        # batch + isolated + summary
    python -m redteam.hardening.audit summarize  # rank analysis (pure Python)

The batch sets GYM_EVAL_WORKERS so that the evaluators run several method
commands at once (one BLAS thread each); scores do not depend on it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
OFFICIAL = [
    "scores.json", "details.json", "scores_train.json", "details_train.json",
    "scores_test.json", "details_test.json",
]
TREES = ["directed_tree_reconstruction", "directed_tree_depth5_reconstruction"]
GRAPHS = ["cora_digraph_reconstruction", "citeseer_digraph_reconstruction"]

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
    "better_weighted": VAR + ["weighted"],
    "better_weighted_seed1": VAR + ["weighted", "--seed-offset", "1"],
    "better_weighted_seed2": VAR + ["weighted", "--seed-offset", "2"],
    "better_multiscale": VAR + ["multiscale"],
    "better_multiscale_seed1": VAR + ["multiscale", "--seed-offset", "1"],
    "better_multiscale_seed2": VAR + ["multiscale", "--seed-offset", "2"],
    "better_alpha_select": VAR + ["alpha_select"],
    "better_long_smacof": ["python", "-m", "method.embed", "--alpha", "0.5", "--max-iter", "2000", "--eps", "1e-10"],
    "weaker_10iter": ["python", "-m", "method.embed", "--alpha", "0.5", "--max-iter", "10"],
    "weaker_5iter": ["python", "-m", "method.embed", "--alpha", "0.5", "--max-iter", "5"],
    "weaker_alpha03": ["python", "-m", "method.embed", "--alpha", "0.3", "--max-iter", "500"],
    "weaker_random_init": VAR + ["random_init"],
    "weaker_isomap_only": VAR + ["isomap_only"],
    "weaker_euclidean": ["python", "-m", "baseline.embed"],
    "probe_level_collapse": VAR + ["level_collapse"],
    "probe_hybrid": VAR + ["hybrid"],
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
    "better_weighted": ["better_weighted", "better_weighted_seed1", "better_weighted_seed2"],
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


def _env(workers: int) -> dict:
    env = dict(os.environ)
    env["GYM_EVAL_WORKERS"] = str(workers)
    if workers > 1:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var] = "1"
    return env


def _workers() -> int:
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(1, min(cpus, 24))


def _run(script: str, log: Path, workers: int = 1) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(["bash", script], cwd=ROOT, stdout=handle,
                              stderr=subprocess.STDOUT, env=_env(workers))
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
    workers = _workers()
    status = {"workers": workers,
              "train": _run("scripts/evaluate_train.sh", dest / "train.log", workers)}
    _collect(dest, ["scores_train.json", "details_train.json"])
    # Forged score file: the evaluator must overwrite it from recomputed metrics.
    forged = json.loads((ROOT / "scoring" / "scores_train.json").read_text())
    for experiment in forged.values():
        for metrics in experiment["results"].values():
            for key in metrics:
                metrics[key] = 0.0 if key in {
                    "normalized_stress", "distortion", "asymmetry_rmse", "symmetric_rmse",
                    "asymmetry_error", "symmetric_error"} else 1.0
    _dump(ROOT / "scoring" / "scores_test.json", forged)
    status["test"] = _run("scripts/evaluate_test.sh", dest / "test.log", workers)
    _collect(dest, ["scores_test.json", "details_test.json"])
    _dump(dest / "exit_status.json", status)
    _clear_cards()


def _patched(path: Path, new_text: str, case: str, card: tuple[str, list[str]] | None) -> None:
    original = path.read_bytes()
    try:
        path.write_text(new_text)
        if card:
            _card(*card)
        dest = RESULTS / case
        _dump(dest / "exit_status.json", {"test": _run("scripts/evaluate_test.sh", dest / "test.log", _workers())})
        _collect(dest, ["scores_test.json", "details_test.json"])
    finally:
        path.write_bytes(original)
        _clear_cards()


def run_isolated() -> None:
    # 1. Replace the public tree generator; the hidden test must not move.
    gen = ROOT / "data" / "directed_tree.py"
    _patched(gen, gen.read_text() + EASY_GENERATOR, "fixed_generator",
             ("shortcut_fixed_generator", SHORT + ["generator_easy"]))

    # 2. Replace the public Cora graph by a trivial chain; the hidden test must not move.
    graph = ROOT / "data" / "citation_graphs" / "cora_lwcc_edges.txt"
    n = int([l for l in graph.read_text().splitlines() if l.startswith("n_nodes")][0].split()[1])
    chain = "n_nodes %d\n" % n + "".join(f"{i + 1} {i}\n" for i in range(n - 1))
    _patched(graph, chain, "public_graph_tamper", None)

    # 3. Sabotage the public baseline; the evaluator's frozen baseline must not move.
    base = ROOT / "baseline" / "euclidean_smacof.py"
    _patched(base, base.read_text() + ZERO_BASELINE, "move_baseline",
             ("shortcut_move_baseline_public", ["python", "-m", "baseline.embed"]))

    # 4. Empty the method lists of the train score file; the test must still
    #    score every method on disk.
    train_scores = ROOT / "scoring" / "scores_train.json"
    saved = json.loads(train_scores.read_text())
    emptied = json.dumps({exp: {"results": {}} for exp in saved}, indent=2)
    _patched(train_scores, emptied, "train_filter", ("probe_level_collapse", VAR + ["level_collapse"]))


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
        if exp not in scores:
            continue
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
    details_train = _load(RESULTS / "batch" / "details_train.json")
    if test is None:
        print("no batch results", file=sys.stderr)
        return
    reference = {exp: test[exp]["results"][REFERENCE] for exp in test}
    methods = sorted(test[next(iter(test))]["results"])
    genuine = [REFERENCE] + [m for m in GENUINE if m in methods]
    attempts = [m for m in methods if m != BASELINE]
    shortcut_names = [m for m in methods if m.startswith("shortcut_")]
    eligible = [m for m in genuine if details and details["methods"].get(m, {}).get("eligible")]

    def ranks_for(method: str, scores: dict, ref: dict) -> dict:
        pool_g = genuine if method in genuine else genuine + [method]
        pool_e = eligible if method in eligible else eligible + [method]
        pool_a = attempts if method in attempts else attempts + [method]
        return {
            "rank_eligible_pool": rank(scores, contract, pool_e, method, ref),
            "rank_genuine_pool": rank(scores, contract, pool_g, method, ref),
            "rank_all_pool": rank(scores, contract, pool_a, method, ref),
        }

    table = {}
    for method in methods:
        info = details["methods"].get(method, {}) if details else {}
        row = {
            "eligible": info.get("eligible"),
            "n_failure_reasons": info.get("n_failure_reasons"),
            "first_failure": (info.get("failure_reasons") or [None])[0],
            **ranks_for(method, test, reference),
            "test": {exp: test[exp]["results"][method] for exp in test},
            "test_raw_mean": {exp: (info.get("experiments", {}).get(exp, {}) or {}).get("raw_mean") for exp in test},
            "runtime_seconds_info_only": (info.get("run") or {}).get("runtime_seconds_info_only"),
        }
        if train and method in train[next(iter(train))]["results"]:
            ref_train = {exp: train[exp]["results"][REFERENCE] for exp in train}
            row["train"] = {exp: train[exp]["results"][method] for exp in train}
            row["train_rank_eligible_pool"] = ranks_for(method, train, ref_train)["rank_eligible_pool"]
            tinfo = details_train["methods"].get(method, {}) if details_train else {}
            row["train_eligible"] = tinfo.get("eligible")
            row["train_raw_mean"] = {exp: (tinfo.get("experiments", {}).get(exp, {}) or {}).get("raw_mean") for exp in train}
        table[method] = row

    seed = {}
    for group, members in SEED_GROUPS.items():
        present = [m for m in members if m in table]
        if not present:
            continue
        for pool in ("rank_eligible_pool", "rank_genuine_pool", "rank_all_pool"):
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
    for group in ("better_exact_smacof", "better_lbfgs", "better_weighted", "better_multiscale"):
        members = SEED_GROUPS[group]
        if not all(m in table for m in members + PAIRED_REFERENCE):
            continue
        for pool in ("rank_eligible_pool", "rank_genuine_pool", "rank_all_pool"):
            diffs = [table[b][pool] - table[r][pool] for b, r in zip(members, PAIRED_REFERENCE)]
            paired.setdefault(group, {})[pool] = {"paired_differences": diffs, "min": min(diffs), "mean": sum(diffs) / len(diffs)}

    # Per-instance paired comparison on the citation experiments (raw metrics).
    per_instance = {}
    if details:
        ref_rows = {exp: details["methods"][REFERENCE]["experiments"][exp]["per_instance"] for exp in GRAPHS if exp in test}
        for method in [BASELINE] + [m for m in genuine if m != REFERENCE]:
            info = details["methods"].get(method)
            if not info:
                continue
            out = {}
            for exp, rrows in ref_rows.items():
                rows = info["experiments"][exp]["per_instance"]
                wins = {"distortion": 0, "map": 0, "normalized_stress": 0}
                n = 0
                for iid, rrow in rrows.items():
                    row = rows.get(iid)
                    if not row or not rrow:
                        continue
                    n += 1
                    wins["distortion"] += row["distortion"] < rrow["distortion"]
                    wins["map"] += row["map"] > rrow["map"]
                    wins["normalized_stress"] += row["normalized_stress"] < rrow["normalized_stress"]
                out[exp] = {"n": n, "instances_better_than_reference": wins}
            per_instance[method] = out

    isolated = {}
    for case in ("fixed_generator", "public_graph_tamper", "move_baseline", "train_filter"):
        scores = _load(RESULTS / case / "scores_test.json")
        status = _load(RESULTS / case / "exit_status.json")
        det = _load(RESULTS / case / "details_test.json")
        isolated[case] = {
            "exit_status": status,
            "methods": sorted(scores[next(iter(scores))]["results"]) if scores else None,
            "eligible": {m: v["eligible"] for m, v in det["methods"].items()} if det else None,
            "reference_unchanged": bool(scores) and all(
                scores[exp]["results"].get(REFERENCE) == test[exp]["results"][REFERENCE] for exp in test),
            "baseline_unchanged": bool(scores) and all(
                scores[exp]["results"].get(BASELINE) == test[exp]["results"][BASELINE] for exp in test),
        }

    forged_overwritten = bool(test) and all(
        test[exp]["results"][REFERENCE]["normalized_stress"] > 0 for exp in test
    )
    summary = {
        "contract": {exp: {"weight": c["weight"], "metrics": {m: list(v) for m, v in c["metrics"].items()}} for exp, c in contract.items()},
        "pools": {"eligible_genuine": eligible, "genuine": genuine, "all_attempts": attempts},
        "methods": table,
        "seed_spread": seed,
        "paired_better_minus_reference": paired,
        "citation_per_instance_vs_reference": per_instance,
        "isolated": isolated,
        "forged_score_file_overwritten": forged_overwritten,
        "shortcuts": shortcut_names,
    }
    _dump(RESULTS / "summary.json", summary)

    def fmt(row, exps, metric, raw=False):
        vals = []
        for e in exps:
            src = (row["test_raw_mean"].get(e) or {}) if raw else row["test"][e]
            vals.append(f"{src[metric]:.4f}" if metric in src else "-")
        return " / ".join(vals)

    lines = [
        "| Attempt | Eligible | Rank E | Rank G | Rank A | Stress d7 / d5 | Distortion d7 / d5 | Distortion cora / citeseer | MAP cora / citeseer | Stress cora / citeseer | Asym. error cora / citeseer | Raw distortion cora / citeseer | Raw MAP cora / citeseer |",
        "|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|",
    ]
    for method in [REFERENCE, BASELINE] + [m for m in methods if m not in (REFERENCE, BASELINE)]:
        row = table[method]
        lines.append(
            f"| {method} | {row['eligible']} | {row['rank_eligible_pool']:+.3f} | {row['rank_genuine_pool']:+.3f} | {row['rank_all_pool']:+.3f} | "
            f"{fmt(row, TREES, 'normalized_stress')} | {fmt(row, TREES, 'distortion')} | {fmt(row, GRAPHS, 'distortion')} | "
            f"{fmt(row, GRAPHS, 'map')} | {fmt(row, GRAPHS, 'normalized_stress')} | {fmt(row, GRAPHS, 'asymmetry_error')} | "
            f"{fmt(row, GRAPHS, 'distortion', raw=True)} | {fmt(row, GRAPHS, 'map', raw=True)} |"
        )
    (RESULTS / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "batch", "isolated", "summarize"])
    args = parser.parse_args()
    backups = {}
    if args.command in ("run", "batch", "isolated"):
        backups = {n: (ROOT / "scoring" / n).read_bytes() for n in OFFICIAL if (ROOT / "scoring" / n).exists()}
    try:
        if args.command in ("run", "batch"):
            run_batch()
        if args.command in ("run", "isolated"):
            run_isolated()
    finally:
        for name, data in backups.items():
            (ROOT / "scoring" / name).write_bytes(data)
    if args.command in ("run", "summarize"):
        summarize()


if __name__ == "__main__":
    main()
