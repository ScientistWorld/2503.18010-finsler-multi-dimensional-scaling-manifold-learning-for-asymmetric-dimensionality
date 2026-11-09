"""Audit-only rank calculation; never imported by a candidate method.

The gym's eventual standard deviations depend on unknown future attempts.
We therefore retain raw scores and report explicitly named finite cohorts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from redteam.audit import CASES, HERE, dump

QUALITY = ['reference_saved', 'reference_seed0', 'reference_seed1', 'reference_seed2',
           'better_long_smacof', 'better_lbfgs', 'weaker_5steps', 'weaker_no_asymmetry']


def load(case):
    path = HERE / case / 'scores_test.json'
    return json.loads(path.read_text()) if path.exists() else None


def proposed(scores):
    return {e: spec['results']['finsler_mds'] for e, spec in scores.items()
            if 'finsler_mds' in spec['results']}


def main():
    contract = json.loads((HERE / 'rank_contract.json').read_text())['experiments']
    raw = {name: load(name) for name in CASES}
    reference = proposed(raw['reference_saved'])
    valid = {}
    for name, scores in raw.items():
        if scores is None:
            continue
        values = proposed(scores)
        if all(e in values and all(m in values[e] and np.isfinite(values[e][m])
                                 for m, spec in exp['metrics'].items() if abs(spec['coefficient']) > 0)
               for e, exp in contract.items()):
            valid[name] = values
    cohorts = {
        'quality_cohort': [n for n in QUALITY if n in valid],
        'all_finite_attempts_cohort': list(valid),
    }
    deviations = {}
    for name, members in cohorts.items():
        deviations[name] = {e: {m: float(np.std([valid[n][e][m] for n in members]))
                               for m, spec in exp['metrics'].items() if abs(spec['coefficient']) > 0}
                            for e, exp in contract.items()}

    def gain(values, sd):
        total = 0.
        parts = {}
        for e, exp in contract.items():
            parts[e] = {}
            for metric, spec in exp['metrics'].items():
                coeff = abs(spec['coefficient'])
                if coeff == 0:
                    continue
                direction = 1 if spec['higher_is_better'] else -1
                delta = direction * (values[e][metric] - reference[e][metric])
                # A metric constant in the cohort contributes zero; use pairwise
                # normalization when a new candidate introduces a new value.
                contribution = 0. if sd[e][metric] == 0 else exp['weight'] * coeff * delta / sd[e][metric]
                total += contribution
                parts[e][metric] = contribution
        return dict(total=total, contributions=parts)

    rows = {}
    for case in CASES:
        row = {'valid_complete_finite_candidate': case in valid}
        if case in valid:
            row['metrics'] = valid[case]
            for cohort, sd in deviations.items():
                row[cohort] = gain(valid[case], sd)
            # The quality cohort cannot normalize a newly changing saturated
            # binary metric. Include this candidate for an honest comparison.
            members = list(dict.fromkeys(cohorts['quality_cohort'] + [case]))
            sd = {e: {m: float(np.std([valid[n][e][m] for n in members]))
                      for m, spec in exp['metrics'].items() if abs(spec['coefficient']) > 0}
                  for e, exp in contract.items()}
            row['quality_plus_this_attempt'] = gain(valid[case], sd)
            deltas = [((1 if spec['higher_is_better'] else -1) *
                       (valid[case][e][m] - reference[e][m]))
                      for e, exp in contract.items() for m, spec in exp['metrics'].items()
                      if abs(spec['coefficient']) > 0]
            row['dominates_reference'] = all(v >= -1e-12 for v in deltas) and any(v > 1e-10 for v in deltas)
            row['ties_reference'] = all(abs(v) <= 1e-12 for v in deltas)
        else:
            row['reason'] = 'Missing candidate/experiments, evaluation failure, or nonfinite scored metrics; no valid rank claimed.'
        rows[case] = row

    seed_cases = ['reference_seed0', 'reference_seed1', 'reference_seed2']
    seed_spread = {}
    for e, exp in contract.items():
        seed_spread[e] = {}
        for m in exp['metrics']:
            vals = [valid[n][e][m] for n in seed_cases]
            seed_spread[e][m] = dict(min=min(vals), max=max(vals), range=max(vals)-min(vals), std=float(np.std(vals)))
    seed_ranks = [rows[n]['quality_cohort']['total'] for n in seed_cases]
    bounds = {'normalized_stress': 0., 'map_at_10': 1., 'asymmetry_rmse': 0.,
              'symmetric_rmse': 0., 'hierarchy_alignment': 1., 'distance_preservation_pass': 1.,
              'asymmetry_preservation_pass': 1., 'hierarchy_direction_pass': 1.}
    saturated = {e: [m for m, b in bounds.items() if abs(reference[e][m]-b) < 1e-12] for e in contract}
    baseline = {e: raw['reference_saved'][e]['results']['euclidean_mds'] for e in contract}
    report = dict(rank_definition='sum(weight * abs(coefficient) * oriented_delta / population_std); reference=0',
                  cohort_members=cohorts, standard_deviations=deviations, attempts=rows,
                  reference=reference, baseline=baseline, saturated_scored_metrics=saturated,
                  seed_metrics=seed_spread, seed_rank_min=min(seed_ranks), seed_rank_max=max(seed_ranks),
                  seed_rank_spread=max(seed_ranks)-min(seed_ranks))
    dump(HERE / 'summary.json', report)
    lines = ['| Attempt | Rank gain (quality cohort + attempt) | Stress d7 / d5 | MAP d7 / d5 | Status |',
             '|---|---:|---|---|---|']
    for case, row in rows.items():
        if not row['valid_complete_finite_candidate']:
            lines.append(f'| {case} | — | — | — | invalid/incomplete |')
            continue
        d7 = row['metrics']['directed_tree_reconstruction']
        d5 = row['metrics']['directed_tree_depth5_reconstruction']
        lines.append(f"| {case} | {row['quality_plus_this_attempt']['total']:+.6f} | "
                     f"{d7['normalized_stress']:.6f} / {d5['normalized_stress']:.6f} | "
                     f"{d7['map_at_10']:.6f} / {d5['map_at_10']:.6f} | "
                     f"{'dominates' if row['dominates_reference'] else 'tie' if row['ties_reference'] else 'tradeoffs'} |")
    (HERE / 'results_table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print('Three-seed rank range:', min(seed_ranks), max(seed_ranks), 'spread:', report['seed_rank_spread'])


if __name__ == '__main__':
    main()
