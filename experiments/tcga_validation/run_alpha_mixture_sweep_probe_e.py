"""Multimodal alpha-mixture sweep for TP53 using probe-score-defined E.

E = top-10% patients by TP53 probe score on A', using each model's own
representation. No cancer-type labels required at any stage.

Output: tp53_probe_e_v3_mix_a{pct}_{model}_split{seed}.json

Usage:
  python experiments/tcga_validation/run_alpha_mixture_sweep_probe_e.py
  python experiments/tcga_validation/run_alpha_mixture_sweep_probe_e.py \\
      --out-dir ./data/tcga
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[2]))
from evaluation.probes import fit_probe, predict_scores

from experiments.tcga_validation.run_decat import (
    _load_cohorts, _safe_evaluate_one, _serialize, _combo_seed,
)

_DEFAULT_OUT_DIR = Path(os.environ.get('DECAT_OUT_DIR', './data/tcga'))
MUTATION_CSV     = os.environ.get('DECAT_MUTATION_CSV', './data/tcga/tcga_driver_mutations.csv')
PROBE_DECILE     = 0.90   # top-10%
N_JOBS           = max(1, os.cpu_count() - 2)
LABEL_TYPE       = 'tp53_probe_e'
N_SEEDS          = 50
ALPHAS           = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Module-level state set by _load_globals()
ho_all: pd.DataFrame | None = None
model_names: list = []


def _load_globals(out_dir: Path) -> None:
    global ho_all, model_names
    repr_dir    = out_dir / 'representations'
    model_names = sorted(d.name for d in repr_dir.iterdir() if d.is_dir())
    meta        = pd.read_csv(out_dir / 'patient_meta.csv')
    splits      = json.load(open(out_dir / 'pooled_random_splits.json'))
    train_set   = set(splits[0]['cohort_a'])
    meta['held_out'] = ~meta['row_idx'].isin(train_set)
    ho_all = meta[meta['held_out']].copy().reset_index(drop=True)
    ts(f"  N_JOBS={N_JOBS}  models={model_names}")
    ts(f"  Held-out patients: {len(ho_all)}")


def make_mixture_splits(ho: pd.DataFrame, extreme_mask: pd.Series,
                            label_type: str) -> list:
    """3-way split: A'(33%), B(33%), C_pool(34%).

    At alpha=0: C random from C_pool. At alpha=1: C from extreme pool E.
    N_C = min(|E ∩ C_pool|, 300) fixed across all alpha.
    """
    SEED_OFFSETS = {'tmb': 0, 'tp53': 1000, 'age': 2000, 'dfs': 3000,
                    'tp53_probe_e': 5000}
    offset = SEED_OFFSETS.get(label_type, 0)
    splits = []
    for seed in range(N_SEEDS):
        rng  = np.random.default_rng(seed + offset + 20000)
        perm = rng.permutation(len(ho))
        n_ap = len(perm) // 3
        n_b  = len(perm) // 3
        ap     = ho.iloc[perm[:n_ap]]
        b      = ho.iloc[perm[n_ap:n_ap + n_b]]
        c_pool = ho.iloc[perm[n_ap + n_b:]]
        y_ap = ap['label'].values.astype(np.int8)
        y_b  = b['label'].values.astype(np.int8)
        c_mask  = extreme_mask.iloc[perm[n_ap + n_b:]].values
        E_in_c  = c_pool[c_mask]['row_idx'].values
        ALL_c   = c_pool['row_idx'].values
        N_C     = min(len(E_in_c), 300)
        if N_C < 20:
            continue
        for alpha in ALPHAS:
            rng_c    = np.random.default_rng(seed + int(alpha * 1000) + offset + 90000)
            n_from_E = int(round(alpha * N_C))
            n_from_R = N_C - n_from_E
            c_parts  = []
            if n_from_E > 0 and len(E_in_c) > 0:
                c_parts.append(rng_c.choice(E_in_c, size=min(n_from_E, len(E_in_c)), replace=False))
            remaining = np.setdiff1d(ALL_c, c_parts[0] if c_parts else np.array([], dtype=int))
            if n_from_R > 0 and len(remaining) > 0:
                c_parts.append(rng_c.choice(remaining, size=min(n_from_R, len(remaining)), replace=False))
            if not c_parts:
                continue
            idx_c = np.concatenate(c_parts)
            y_c   = ho.set_index('row_idx').loc[idx_c, 'label'].values.astype(np.int8)
            splits.append({
                'seed':   seed,
                'alpha':  alpha,
                'idx_ap': ap['row_idx'].values,
                'idx_b':  b['row_idx'].values,
                'idx_c':  idx_c,
                'y_ap':   y_ap,
                'y_b':    y_b,
                'y_c':    y_c,
            })
    return splits


def ts(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_probe_e_mask(model_name: str, ho: pd.DataFrame,
                       repr_dir: Path, splits_json: list) -> pd.Series:
    """Train TP53 probe on A' using model_name's H&E representation.

    Returns mask_probe_s: pd.Series (bool, indexed like ho),
    E = top-PROBE_DECILE scoring patients.
    """
    # Use split-0 A' cohort
    ap_ridx = np.array(splits_json[0]['cohort_aprime'])
    ho_ridx = ho['row_idx'].values

    # Get A' indices that are in ho
    ap_in_ho  = np.where(np.isin(ho_ridx, ap_ridx))[0]
    y_ap      = ho['label'].values[ap_in_ho].astype(np.int8)

    # Load A' H&E cohort
    dummy_sp = {
        'idx_ap': ho_ridx[ap_in_ho],
        'idx_b':  ho_ridx[ap_in_ho[:10]],   # dummy, unused
        'idx_c':  ho_ridx[ap_in_ho[:10]],   # dummy, unused
    }
    Zh_ap, _, _ = _load_cohorts(repr_dir, model_name,
                                dummy_sp['idx_ap'],
                                dummy_sp['idx_b'],
                                dummy_sp['idx_c'])
    Z_ap   = Zh_ap['Ap']
    valid  = ~np.isnan(Z_ap[:, 0]) & (y_ap >= 0)
    Z_v    = Z_ap[valid]; y_v = y_ap[valid]

    sc, probe = fit_probe(Z_v, y_v, probe_backend='logistic',
                          C=0.1, solver='liblinear', max_iter=500, random_state=42)
    ts(f"    Probe on A': N={len(y_v)}, prev={y_v.mean():.1%}")

    # Score all held-out patients
    scores = np.full(len(ho), -np.inf)
    # Load all held-out H&E representations using _load_cohorts with full ho as "Ap"
    Zh_all, _, _ = _load_cohorts(repr_dir, model_name,
                                 ho_ridx,
                                 ho_ridx[:10],   # dummy
                                 ho_ridx[:10])   # dummy
    Z_all  = Zh_all['Ap']
    valid2 = ~np.isnan(Z_all[:, 0])
    scores[valid2] = predict_scores(sc, probe, Z_all[valid2])

    thresh      = np.percentile(scores[scores > -np.inf], PROBE_DECILE * 100)
    mask_probe  = scores >= thresh

    # Log cancer-type composition
    ct_comp = ho[mask_probe].groupby('cancer_type').size().sort_values(ascending=False).head(5)
    ts(f"    Probe-score E: |E|={mask_probe.sum()}, top CTs: {dict(ct_comp)}")

    return pd.Series(mask_probe, index=ho.index)


def run_model(model_name: str, ho: pd.DataFrame, repr_dir: Path,
              res_dir: Path, splits_json: list) -> None:
    """Run full alpha-mixture sweep for TP53 × model_name using probe-score E."""
    ts(f"\n  Model: {model_name}")

    # Build probe-score E for this model
    mask_probe_s = build_probe_e_mask(model_name, ho, repr_dir, splits_json)

    # Generate splits using this model's probe-score E
    splits = make_mixture_splits(ho, mask_probe_s, LABEL_TYPE)
    ts(f"    {len(splits)} splits ({N_SEEDS} seeds × {len(ALPHAS)} alphas)")

    n_done = 0
    for sp in splits:
        alpha = sp['alpha']
        seed  = sp['seed']
        out_path = res_dir / f"{LABEL_TYPE}_v3_mix_a{int(alpha*100):03d}_{model_name}_split{seed}.json"

        if out_path.exists():
            n_done += 1
            continue

        # Load cohorts for this model only
        Zh, Zr, gamma = _load_cohorts(repr_dir, model_name,
                                      sp['idx_ap'], sp['idx_b'], sp['idx_c'])
        is_fac = gamma is not None

        jobs = [(
            seed, f'{LABEL_TYPE}_v2_mix_a{alpha:.2f}', model_name, is_fac, gamma,
            {'Ap': Zh['Ap'], 'B': Zh['B'], 'C': Zh['C']},
            {'Ap': Zr['Ap'], 'B': Zr['B'], 'C': Zr['C']},
            sp['y_ap'], sp['y_b'], sp['y_c'], LABEL_TYPE, 0,
            _combo_seed(seed, f'{LABEL_TYPE}_v2_mix', model_name, LABEL_TYPE, int(alpha * 100)),
        )]

        batches = Parallel(n_jobs=N_JOBS, backend='loky')(
            delayed(_safe_evaluate_one)(*j) for j in jobs)
        records = [r for b in batches for r in b]

        with open(out_path, 'w') as f:
            json.dump([_serialize(r) for r in records], f)

    # Summary
    ts(f"\n    {model_name} TP53 probe-E — S2 rate by alpha:")
    print(f"    {'alpha':>6}  {'S1':>6}  {'S2':>6}  {'Ind':>6}  {'S3':>6}")
    for alpha in ALPHAS:
        recs = []
        for fp in glob.glob(str(res_dir / f"{LABEL_TYPE}_v3_mix_a{int(alpha*100):03d}_{model_name}_split*.json")):
            recs.extend(json.load(open(fp)))
        recs = [r for r in recs if r.get('modality') == 'h']
        if not recs: continue
        n  = len(recs)
        s1 = sum(r['predicted_scenario'] == 'scenario_1'    for r in recs) / n
        s2 = sum(r['predicted_scenario'] == 'scenario_2'    for r in recs) / n
        ind= sum(r['predicted_scenario'] == 'indeterminate' for r in recs) / n
        s3 = sum(r['predicted_scenario'] == 'scenario_3'    for r in recs) / n
        print(f"    {alpha:>6.1f}  {s1:>6.2f}  {s2:>6.2f}  {ind:>6.2f}  {s3:>6.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument('--models', nargs='+', default=None,
                        metavar='MODEL',
                        help='Model keys to evaluate (default: all 7). '
                             'E.g. --models clip_K50 cca_K50')
    args = parser.parse_args()

    out_dir  = args.out_dir
    res_dir  = out_dir / 'results'
    repr_dir = out_dir / 'representations'
    res_dir.mkdir(parents=True, exist_ok=True)

    ts("=== TP53 alpha-mixture sweep — probe-score E (multimodal) ===")
    ts(f"  OUT_DIR     = {out_dir}")
    ts(f"  LABEL_TYPE  = {LABEL_TYPE}  (output files: {LABEL_TYPE}_v3_mix_a*_<model>_split*.json)")
    ts(f"  ALPHAS      = {len(ALPHAS)} values: {ALPHAS}")
    ts(f"  E defn      = top-{(1-PROBE_DECILE)*100:.0f}% probe scores per model (no cancer-type labels)")

    _load_globals(out_dir)

    meta        = pd.read_csv(out_dir / 'patient_meta.csv')
    muts        = pd.read_csv(MUTATION_CSV).set_index('patient_id')
    splits_json = json.load(open(out_dir / 'pooled_random_splits.json'))[:N_SEEDS]

    # TP53 label
    ho = ho_all.copy()
    ho['label'] = ho['patient_id'].map(muts['TP53'])
    ho = ho[ho['label'].notna()].reset_index(drop=True)
    ho['label'] = ho['label'].astype(np.int8)
    ts(f"\n  TP53: N={len(ho)}, prev={ho['label'].mean():.1%}")

    models_to_run = args.models if args.models else model_names
    ts(f"  Models: {models_to_run}")

    for model_name in models_to_run:
        run_model(model_name, ho, repr_dir, res_dir, splits_json)

    ts("\n=== Done ===")


if __name__ == '__main__':
    main()
