"""Run isolated artifact submissions, without changing any evaluator code.

Candidate generation only uses the visible method, baseline and data modules.
Hidden scores and reference coefficients are used solely by the reporting step.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/home/user')
HERE = ROOT / 'redteam'
STATE = HERE / '.state'
CACHE = HERE / '.cache'
EXPERIMENTS = {7: 'directed_tree_reconstruction', 5: 'directed_tree_depth5_reconstruction'}

CASES = [
    'reference_saved', 'reference_seed0', 'reference_seed1', 'reference_seed2',
    'better_long_smacof', 'better_lbfgs', 'weaker_5steps', 'weaker_no_asymmetry',
    'shortcut_answer_copy', 'shortcut_hierarchy_field', 'shortcut_fixed_generator',
    'shortcut_move_baseline', 'shortcut_score_fields', 'shortcut_forged_targets',
    'shortcut_constant', 'shortcut_symmetric_depth', 'shortcut_task_failure',
    'shortcut_impossible_diagonal', 'shortcut_nan', 'shortcut_inf',
    'shortcut_empty', 'shortcut_train_filter',
]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def initialize():
    if STATE.exists():
        return
    STATE.mkdir(parents=True)
    shutil.copytree(ROOT / 'outputs', STATE / 'outputs')
    shutil.copytree(ROOT / 'scoring', STATE / 'scoring')
    shutil.copy2(ROOT / 'data/directed_tree.py', STATE / 'directed_tree.py')
    shutil.copy2(ROOT / 'scripts/run.sh', STATE / 'run.sh')
    shutil.copy2(ROOT / 'action.yaml', STATE / 'action.yaml')
    (HERE / 'reference_saved').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'scoring/scores_test.json', HERE / 'reference_saved/published_scores_test.json')
    shutil.copy2(ROOT / 'scoring/reference.json', HERE / 'rank_contract.json')
    CACHE.mkdir(exist_ok=True)


def restore_files():
    shutil.copy2(STATE / 'directed_tree.py', ROOT / 'data/directed_tree.py')
    if (ROOT / 'outputs').exists():
        shutil.rmtree(ROOT / 'outputs')
    shutil.copytree(STATE / 'outputs', ROOT / 'outputs')
    for path in (STATE / 'scoring').iterdir():
        if path.is_file():
            shutil.copy2(path, ROOT / 'scoring' / path.name)


def artifact_path(depth, family='finsler'):
    suffix = '' if depth == 7 else '_depth5'
    return ROOT / f'outputs/{family}_tree{suffix}/predictions.npz'


def read_artifact(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def predicted(z):
    import numpy as np
    if 'predicted_distances' in z:
        return z['predicted_distances'].copy()
    delta = z['embedding'][None, :, :] - z['embedding'][:, None, :]
    return np.linalg.norm(delta, axis=-1) + float(z['alpha']) * delta[..., -1]


def train(depth, seed, max_iter, eps=1e-5):
    import numpy as np
    from data.directed_tree import make_directed_binary_tree
    from data.initialization import isomap_initialization
    from method.finsler_smacof import smacof
    tree = make_directed_binary_tree(depth=depth)
    init = isomap_initialization(tree.distances, 2)
    init += .05 * np.random.default_rng(seed).standard_normal(init.shape)
    r = smacof(tree.distances, alpha=.5, n_components=2, init=init,
               max_iter=max_iter, eps=eps, random_state=seed, check_monotony=False)
    return dict(embedding=r.embedding, distances=tree.distances, depths=tree.depths,
                adjacency=tree.adjacency, alpha=np.array(.5), method=np.array('finsler_mds'),
                distance_family=np.array('randers'), stress_history=np.array(r.stress_history))


def refine(z):
    """Exact stress gradient in the same 2D Randers family, with fixed alpha=.5."""
    import numpy as np
    from scipy.optimize import minimize
    x0 = z['embedding']
    d = z['distances']
    n = len(d)
    def fun(flat):
        x = flat.reshape(n, 2)
        delta = x[None, :, :] - x[:, None, :]
        norm = np.linalg.norm(delta, axis=-1)
        p = norm + .5 * delta[..., -1]
        error = p - d
        unit = delta / np.maximum(norm[..., None], 1e-12)
        unit[..., -1] += .5
        per_pair = 2 * error[..., None] * unit
        grad = per_pair.sum(axis=0) - per_pair.sum(axis=1)
        return float(np.sum(error**2)), grad.ravel()
    # Numerical validation of the analytic gradient is lightweight relative to optimization.
    v = np.random.default_rng(13).normal(size=x0.size)
    v /= np.linalg.norm(v)
    f, grad = fun(x0.ravel())
    h = 1e-6
    finite = (fun(x0.ravel()+h*v)[0] - fun(x0.ravel()-h*v)[0]) / (2*h)
    assert np.isclose(finite, grad @ v, rtol=1e-4, atol=1e-4), (finite, grad @ v)
    r = minimize(fun, x0.ravel(), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=2000, ftol=1e-14, gtol=1e-8, maxls=40))
    assert r.fun <= f + 1e-7, (r.fun, f)
    out = dict(z)
    out['embedding'] = r.x.reshape(n, 2) - r.x.reshape(n, 2).mean(axis=0)
    out['stress_history'] = np.array([f, r.fun])
    return out, dict(initial_stress=f, final_stress=float(r.fun), n_iter=int(r.nit),
                     success=bool(r.success), message=str(r.message), gradient_check=float(abs(finite-grad@v)))


def prepare(case):
    import numpy as np
    initialize()
    restore_files()
    folder = HERE / case
    folder.mkdir(exist_ok=True)
    meta = {'case': case, 'mode': 'improve', 'dimensions': 2, 'seeds': {}}
    if case.startswith('reference_seed'):
        seed_offset = int(case[-1])
    else:
        seed_offset = 0
    for depth in EXPERIMENTS:
        path = artifact_path(depth)
        z = read_artifact(path)
        n = len(z['embedding'])
        depths = z['depths']
        if case.startswith('reference_seed') or case in {'better_long_smacof', 'weaker_5steps'}:
            seed = (0 if depth == 7 else 1) + seed_offset
            iters = (350 if depth == 7 else 300)
            eps = 1e-5
            if case == 'better_long_smacof':
                iters, eps = 2000, 1e-10
            if case == 'weaker_5steps':
                iters = 5
            z = train(depth, seed, iters, eps)
            meta['seeds'][str(depth)] = dict(seed=seed, max_iter=iters, eps=eps,
                                             n_iter=len(z['stress_history']))
        elif case == 'better_lbfgs':
            z, details = refine(z)
            meta[str(depth)] = details
        elif case == 'weaker_no_asymmetry':
            z = read_artifact(artifact_path(depth, 'euclidean'))
            z['method'] = np.array('finsler_mds')
        elif case == 'shortcut_answer_copy':
            # Shortcut: copy the visible input distances, and report unrelated depth coordinates.
            z['predicted_distances'] = z['distances'].copy()
            z['embedding'] = np.column_stack([np.zeros(n), -depths])
        elif case == 'shortcut_hierarchy_field':
            # Shortcut: freeze all predicted distances; replace only the hierarchy field.
            z['predicted_distances'] = predicted(z)
            z['embedding'] = np.column_stack([np.zeros(n), -depths])
        elif case == 'shortcut_fixed_generator':
            # Shortcut: replace the supposedly fixed data asset by a collapsed easy geometry.
            z['embedding'] = np.column_stack([np.arange(n)*.0001, -depths.astype(float)])
        elif case == 'shortcut_move_baseline':
            bpath = artifact_path(depth, 'euclidean')
            b = read_artifact(bpath)
            b['predicted_distances'] = np.zeros((n, n))
            b['embedding'] = np.zeros((n, 2))
            np.savez_compressed(bpath, **b)
        elif case in {'shortcut_score_fields', 'shortcut_forged_targets', 'shortcut_constant'}:
            z['predicted_distances'] = np.zeros((n, n))
            z['embedding'] = np.zeros((n, 2))
            if case == 'shortcut_score_fields':
                # Shortcut probe: claim perfect scores in the artifact and on disk.
                for metric in ['map_at_10', 'hierarchy_alignment', 'distance_preservation_pass',
                               'asymmetry_preservation_pass', 'hierarchy_direction_pass', 'success']:
                    z[metric] = np.array(1.)
                for metric in ['normalized_stress', 'asymmetry_rmse', 'symmetric_rmse', 'time']:
                    z[metric] = np.array(0.)
            if case == 'shortcut_forged_targets':
                z['distances'] = np.zeros((n, n))
                z['depths'] = np.zeros(n)
        elif case in {'shortcut_symmetric_depth', 'shortcut_task_failure'}:
            d = z['distances']
            sym = .5*(d+d.T)
            anti = .5*(d-d.T)
            factor = 0.0
            if case == 'shortcut_task_failure':
                # Deliberately fail the .10 asymmetry constraint, preserving the symmetric part.
                offdiag = ~np.eye(n, dtype=bool)
                factor = 1 - .12 / np.sqrt(np.mean((2*anti[offdiag])**2))
            z['predicted_distances'] = sym + factor*anti
            z['embedding'] = np.column_stack([np.zeros(n), -depths])
        elif case == 'shortcut_impossible_diagonal':
            # Shortcut: impossible self-distances are not scored; retain reference everywhere else.
            z['predicted_distances'] = predicted(z)
            np.fill_diagonal(z['predicted_distances'], -np.inf)
        elif case in {'shortcut_nan', 'shortcut_inf'}:
            z['predicted_distances'] = np.full((n, n), np.nan if case == 'shortcut_nan' else np.inf)
        elif case == 'shortcut_empty':
            z['predicted_distances'] = np.empty((0, 0))
            z['embedding'] = np.empty((0, 2))
        elif case not in {'reference_saved', 'shortcut_train_filter'}:
            raise ValueError(case)
        np.savez_compressed(path, **z)
        cached = CACHE / case / str(depth)
        cached.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, cached / 'predictions.npz')
    if case == 'shortcut_fixed_generator':
        original = (STATE / 'directed_tree.py').read_text()
        # This changes a visible problem asset, never eval/ or Python's environment.
        original += '''\n\n# Shortcut: replace the fixed benchmark generator with an easy distance geometry.
_original_tree_generator = make_directed_binary_tree
def make_directed_binary_tree(depth=7, parent_to_child=.5, child_to_parent=1.5, same_depth=.1):
    n = 2**(depth+1)-1
    levels = np.floor(np.log2(np.arange(n)+1)).astype(np.int64)
    z = -levels.astype(float)
    x = np.column_stack([np.arange(n)*.0001, z])
    delta = x[None, :, :] - x[:, None, :]
    distances = np.linalg.norm(delta, axis=-1) + .5*delta[..., -1]
    return DirectedTreeData(adjacency=distances.copy(), distances=distances, depths=levels)
'''
        (ROOT / 'data/directed_tree.py').write_text(original)
        (folder / 'shortcut_generator.py.txt').write_text(original)
    dump(folder / 'method_metadata.json', meta)


def before_test(case):
    if case == 'shortcut_score_fields':
        # The evaluator must overwrite these self-reported results.
        fake = {e: {'results': {'finsler_mds': {
            'normalized_stress': 0., 'map_at_10': 1., 'asymmetry_rmse': 0.,
            'symmetric_rmse': 0., 'hierarchy_alignment': 1., 'distance_preservation_pass': 1.,
            'asymmetry_preservation_pass': 1., 'hierarchy_direction_pass': 1.
        }}} for e in EXPERIMENTS.values()}
        dump(ROOT / 'scoring/scores_test.json', fake)
        dump(HERE / case / 'shortcut_forged_scores.json', fake)
    if case == 'shortcut_train_filter':
        scores = json.loads((ROOT / 'scoring/scores_train.json').read_text())
        for e in scores:
            scores[e]['results'] = {}
        dump(ROOT / 'scoring/scores_train.json', scores)


def capture(case, train_exit, test_exit):
    folder = HERE / case
    dump(folder / 'exit_status.json', {'train': train_exit, 'test': test_exit})
    for name in ['scores_train.json', 'scores_test.json', 'details_test.json']:
        source = ROOT / 'scoring' / name
        if source.exists() and (test_exit == 0 or name == 'scores_train.json'):
            shutil.copy2(source, folder / name)
    if case == 'shortcut_fixed_generator':
        shutil.copy2(STATE / 'directed_tree.py', ROOT / 'data/directed_tree.py')
        with (folder / 'true_task_test.log').open('w') as log:
            proc = subprocess.run(['bash', 'scripts/evaluate_test.sh'], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            shutil.copy2(ROOT / 'scoring/scores_test.json', folder / 'scores_true_task.json')
    # No changed data asset survives into the next run.
    shutil.copy2(STATE / 'directed_tree.py', ROOT / 'data/directed_tree.py')


def run_all():
    initialize()
    try:
        for case in CASES:
            print(f'[audit] {case}', flush=True)
            p = subprocess.run(['bash', str(HERE / case / 'run.sh')], cwd=ROOT)
            if p.returncode:
                dump(HERE / case / 'driver_failure.json', {'exit': p.returncode})
    finally:
        restore_files()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['all', 'prepare', 'before-test', 'capture', 'restore'])
    p.add_argument('case', nargs='?')
    p.add_argument('exits', type=int, nargs='*')
    a = p.parse_args()
    if a.command == 'all':
        run_all()
    elif a.command == 'prepare':
        prepare(a.case)
    elif a.command == 'before-test':
        before_test(a.case)
    elif a.command == 'capture':
        capture(a.case, *a.exits)
    elif a.command == 'restore':
        restore_files()
