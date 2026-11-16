#!/usr/bin/env python
"""Hardening audit: replay genuine variants and shortcuts through the hardened gym.

Every candidate is installed only as a method card under ``outputs/audit/`` and
scored by the unchanged ``bash scripts/evaluate_train.sh`` /
``bash scripts/evaluate_test.sh`` (no arguments). Shortcuts that modify
problem files (the public tree generator, the public citation graphs, the
public baseline, the train score file) run in isolation and are restored
afterwards. The official score files live in ``results/official/`` and are
copied back into ``scoring/`` after the audit, so the audit never changes them.

The scheduler may restart a job at any time, so every step is resumable:

    python -m redteam.hardening.audit recover     # undo an interrupted step
    python -m redteam.hardening.audit step paper  # official reproduction steps
    python -m redteam.hardening.audit step train  #   (skipped when already done
    python -m redteam.hardening.audit step test   #    for the current code)
    python -m redteam.hardening.audit run         # chunked batch + isolated + summary
    python -m redteam.hardening.audit summarize   # rank analysis (pure Python)

Each step and each audit chunk stores the SHA-256 of the code it ran
(evaluators, graphs, method, baseline, data, audit variants and shortcuts,
scripts) and is skipped on a restart only if that hash still matches. Files a
step tampers with are first backed up under ``results/state/pending/`` and
restored by ``recover`` if the step was killed.

The batch runs the cards in chunks of at most ``CHUNK`` cards and sets
GYM_EVAL_WORKERS so that the evaluators run them at once (one BLAS thread
each); scores do not depend on either. The reference card
(``outputs/finsler_mds``) and the frozen baseline are scored in every chunk;
the merge checks that they agree across chunks.
"""

from __future__ import annotations

import argparse
import hashlib
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
OFFICIAL_DIR = RESULTS / "official"
STATE = RESULTS / "state"
PENDING = STATE / "pending"
AUDIT_CARDS = ROOT / "outputs" / "audit"
REFERENCE = "finsler_mds"
BASELINE = "euclidean_mds"
OFFICIAL = [
    "scores.json", "details.json", "scores_train.json", "details_train.json",
    "scores_test.json", "details_test.json",
]
STEPS = {
    "paper": ("scripts/method.sh", ["scores.json", "details.json"]),
    "train": ("scripts/evaluate_train.sh", ["scores_train.json", "details_train.json"]),
    "test": ("scripts/evaluate_test.sh", ["scores_test.json", "details_test.json"]),
}
TREES = ["directed_tree_reconstruction", "directed_tree_depth5_reconstruction"]
GRAPHS = ["cora_digraph_reconstruction", "citeseer_digraph_reconstruction"]
CHUNK = 8

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


# Rough relative cost (seconds on the visible slice) used only to group the
# slow cards together, so that each chunk's wall time stays short.
COST = {
    "better_exact_smacof": 450, "better_weighted": 420, "better_lbfgs": 330,
    "better_multiscale": 300, "better_long_smacof": 200, "better_alpha_select": 125,
    "weaker_euclidean": 85, "weaker_random_init": 55, "shortcut_wrong_dim": 45,
}


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


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


# --------------------------------------------------------------------------
# Restart safety
# --------------------------------------------------------------------------


def code_stamp(audit: bool = True) -> str:
    """SHA-256 over every file that can change a score.

    ``audit=False`` covers what the official reproduction depends on
    (evaluators, graphs, method, baseline, data, scripts); ``audit=True`` adds
    the audit's variants and shortcuts.
    """

    files: list[Path] = []
    for folder in ("eval", "method", "baseline", "data"):
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or "downloads" in path.parts:
                continue
            if path.suffix in {".py", ".txt", ".json"}:
                files.append(path)
    if audit:
        files += [HERE / "variants.py", HERE / "shortcuts.py"]
    files += [ROOT / "scripts" / s for s in ("method.sh", "evaluate.sh", "evaluate_train.sh", "evaluate_test.sh")]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _key(*parts) -> str:
    audit = parts[0] != "step"
    return hashlib.sha256(json.dumps([code_stamp(audit), *parts], sort_keys=True).encode()).hexdigest()


def _done(folder: Path, key: str) -> bool:
    record = _load(folder / "done.json")
    return bool(record) and record.get("key") == key


def _tamper(path: Path, new_text: str) -> None:
    """Back the file up under results/state/pending before modifying it."""

    PENDING.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    backup = PENDING / f"{tag}.orig"
    if not backup.exists():
        shutil.copy2(path, backup)
        _dump(PENDING / f"{tag}.json", {"path": str(path.relative_to(ROOT))})
    path.write_text(new_text)


def _untamper_all() -> list[str]:
    restored = []
    if not PENDING.exists():
        return restored
    for record_path in sorted(PENDING.glob("*.json")):
        record = json.loads(record_path.read_text())
        backup = record_path.with_suffix(".orig")
        if backup.exists():
            shutil.copy2(backup, ROOT / record["path"])
            backup.unlink()
            restored.append(record["path"])
        record_path.unlink()
    return restored


def _restore_official() -> list[str]:
    restored = []
    for step, (_, names) in STEPS.items():
        if not _done(OFFICIAL_DIR / step, _key("step", step)):
            continue
        for name in names:
            src = OFFICIAL_DIR / name
            if src.exists():
                shutil.copy2(src, ROOT / "scoring" / name)
                restored.append(name)
    return restored


def recover() -> None:
    _clear_cards()
    print("recover: restored tampered files", _untamper_all(), flush=True)
    print("recover: restored official score files", _restore_official(), flush=True)


def run_step(step: str) -> None:
    """One official reproduction step, skipped if already done for this code."""

    script, names = STEPS[step]
    folder = OFFICIAL_DIR / step
    key = _key("step", step)
    if _done(folder, key):
        print(f"step {step}: already done for this code, restoring its score files", flush=True)
        for name in names:
            shutil.copy2(OFFICIAL_DIR / name, ROOT / "scoring" / name)
        return
    _clear_cards()
    code = subprocess.run(["bash", script], cwd=ROOT).returncode
    if code != 0:
        raise SystemExit(f"step {step}: {script} exited with {code}")
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(ROOT / "scoring" / name, OFFICIAL_DIR / name)
    _dump(folder / "done.json", {"key": key, "script": script, "files": names})
    print(f"step {step}: done", flush=True)


# --------------------------------------------------------------------------
# Batch (chunked, resumable)
# --------------------------------------------------------------------------


def _batch_cards() -> list[tuple[str, list[str] | None, str | None]]:
    cards = [(name, command, None) for name, command in GENUINE.items()]
    cards += [(f"shortcut_{kind}", SHORT + [kind], None) for kind in BATCH_SHORTCUTS]
    cards.append(("shortcut_bad_card", None, '{"name": "shortcut_bad_card", "command": "python -c 1"'))

    def cost(card):
        name = card[0]
        for prefix, value in COST.items():
            if name.startswith(prefix):
                return value
        return 20

    return sorted(cards, key=lambda card: (-cost(card), card[0]))


def _forged_scores() -> dict:
    template = _load(OFFICIAL_DIR / "scores_test.json") or {}
    forged = json.loads(json.dumps(template))
    for experiment in forged.values():
        for metrics in experiment["results"].values():
            for key in metrics:
                metrics[key] = 0.0 if key in {
                    "normalized_stress", "distortion", "asymmetry_rmse", "symmetric_rmse",
                    "asymmetry_error", "symmetric_error"} else 1.0
    return forged


def _merge(slice_name: str, chunk_dirs: list[Path]) -> dict:
    scores: dict = {}
    details: dict = {}
    consistency = {REFERENCE: True, BASELINE: True}
    for folder in chunk_dirs:
        s = _load(folder / f"scores_{slice_name}.json")
        d = _load(folder / f"details_{slice_name}.json")
        if not details:
            details = {k: v for k, v in d.items() if k != "methods"}
            details["methods"] = {}
        for name, info in d["methods"].items():
            if name in details["methods"]:
                if name in consistency:
                    same = all(
                        scores[exp]["results"][name] == s[exp]["results"][name] for exp in s
                    )
                    consistency[name] = consistency[name] and same
                continue
            details["methods"][name] = info
            for exp in s:
                scores.setdefault(exp, {"results": {}})["results"][name] = s[exp]["results"][name]
    dest = RESULTS / "batch"
    _dump(dest / f"scores_{slice_name}.json", scores)
    _dump(dest / f"details_{slice_name}.json", details)
    return consistency


def run_batch() -> None:
    cards = _batch_cards()
    chunks = [cards[i : i + CHUNK] for i in range(0, len(cards), CHUNK)]
    status: dict = {"chunk_size": CHUNK, "chunks": {}}
    for slice_name in ("train", "test"):
        chunk_dirs = []
        for k, chunk in enumerate(chunks):
            folder = RESULTS / "batch" / "chunks" / f"{slice_name}_{k:02d}"
            chunk_dirs.append(folder)
            key = _key("batch", slice_name, chunk)
            if _done(folder, key):
                print(f"batch {slice_name} chunk {k}: already done", flush=True)
                status["chunks"][folder.name] = _load(folder / "done.json")
                continue
            _clear_cards()
            for name, command, raw in chunk:
                _card(name, command, raw)
            forged_written = False
            if slice_name == "test":
                # The evaluator must overwrite a forged score file from recomputed metrics.
                _dump(ROOT / "scoring" / "scores_test.json", _forged_scores())
                forged_written = True
            script = f"scripts/evaluate_{slice_name}.sh"
            print(f"batch {slice_name} chunk {k}: {[c[0] for c in chunk]}", flush=True)
            code = _run(script, folder / f"{slice_name}.log", min(_workers(), len(chunk) + 1))
            _collect(folder, [f"scores_{slice_name}.json", f"details_{slice_name}.json"])
            record = {"key": key, "exit": code, "cards": [c[0] for c in chunk],
                      "forged_file_written_before": forged_written}
            _dump(folder / "done.json", record)
            status["chunks"][folder.name] = record
            _clear_cards()
        status[f"{slice_name}_consistency"] = _merge(slice_name, chunk_dirs)
    _dump(RESULTS / "batch" / "exit_status.json", status)


# --------------------------------------------------------------------------
# Isolated cases (each resumable)
# --------------------------------------------------------------------------


def _patched(path: Path, new_text: str, case: str, card: tuple[str, list[str]] | None) -> None:
    dest = RESULTS / case
    key = _key("isolated", case, hashlib.sha256(new_text.encode()).hexdigest(), card)
    if _done(dest, key):
        print(f"isolated {case}: already done", flush=True)
        return
    try:
        _clear_cards()
        _tamper(path, new_text)
        if card:
            _card(*card)
        code = _run("scripts/evaluate_test.sh", dest / "test.log", _workers())
        _collect(dest, ["scores_test.json", "details_test.json"])
        _dump(dest / "exit_status.json", {"test": code})
    finally:
        _untamper_all()
        _clear_cards()
    _dump(dest / "done.json", {"key": key})
    print(f"isolated {case}: done", flush=True)


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
    saved = _load(OFFICIAL_DIR / "scores_train.json")
    emptied = json.dumps({exp: {"results": {}} for exp in saved}, indent=2)
    _patched(ROOT / "scoring" / "scores_train.json", emptied, "train_filter",
             ("probe_level_collapse", VAR + ["level_collapse"]))


# --------------------------------------------------------------------------
# Rank analysis (pure Python so it also runs on the login node)
# --------------------------------------------------------------------------


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
    parser.add_argument("command", choices=["recover", "step", "run", "batch", "isolated", "summarize", "stamp"])
    parser.add_argument("step", nargs="?", choices=list(STEPS))
    args = parser.parse_args()
    if args.command == "stamp":
        print("official", code_stamp(audit=False))
        print("audit   ", code_stamp(audit=True))
        return
    if args.command == "recover":
        recover()
        return
    if args.command == "step":
        if not args.step:
            parser.error("step needs one of " + ", ".join(STEPS))
        recover()
        run_step(args.step)
        return
    if args.command in ("run", "batch", "isolated"):
        missing = [s for s in STEPS if not _done(OFFICIAL_DIR / s, _key("step", s))]
        if missing:
            raise SystemExit(f"official steps not done for this code: {missing}")
        recover()
        try:
            if args.command in ("run", "batch"):
                run_batch()
            if args.command in ("run", "isolated"):
                run_isolated()
        finally:
            recover()
    if args.command in ("run", "summarize"):
        summarize()


if __name__ == "__main__":
    main()
