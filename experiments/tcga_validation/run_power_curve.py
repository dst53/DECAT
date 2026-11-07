"""Power curve analysis: S2 detection rate vs cohort size N.

Two designs answering complementary practitioner questions:

  Option A — equal_N (A'=B=C=N varied together):
    "How many patients total do I need per cohort?"
    All three cohorts shrink together — worst case, represents a small
    prospective study where everything is resource-limited.

  Option B — vary_C (A'/B fixed at full size, only C varied):
    "I have a well-powered reference cohort; how small can my deployment
    cohort be and still detect confounding?"
    Isolates D_quantile's Wasserstein power from probe quality.

The gap between the two curves shows the value of a large reference cohort
(A'/B): when probe quality is held constant, confounding is detectable at
much smaller C than the equal-N design suggests.

Labels: TMB (E = top-25% by log-TMB), TP53 (E = cancer types with prevalence >0.65 or <0.15),
        Age (E = top-25% by age)
50 seeds per (N, label, design).

Extreme pool E (label-extreme patients) and balanced pool (non-extreme patients)
definitions match run_alpha_mixture_sweep.py — same seed offset convention.

Results:
  <OUT_DIR>/results/{label}_power_equalN_N{N}_split{seed}.json
  <OUT_DIR>/results/{label}_power_varyC_N{N}_split{seed}.json

Usage:
  python experiments/tcga_validation/run_power_curve.py
  python experiments/tcga_validation/run_power_curve.py --seed-only 0
"""

import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parents[2]))
from experiments.tcga_validation.run_decat import (
    _safe_evaluate_one, _load_cohorts, _combo_seed, _serialize,
)

OUT_DIR  = Path(os.environ.get('DECAT_OUT_DIR', './data/tcga'))  # override with --out-dir
N_SEEDS  = 50
N_VALS   = [25, 50, 100, 150, 200, 300, 500, 750, 1000]
N_JOBS   = max(1, os.cpu_count() - 2)

CLINICAL_CSV  = os.environ.get('DECAT_CLINICAL_CSV', './data/tcga/CLINICAL.csv')
TMB_CSV       = os.environ.get('DECAT_TMB_CSV', './data/tcga/tcga_tmb.csv')
MUTATION_CSV  = os.environ.get('DECAT_MUTATION_CSV', './data/tcga/tcga_driver_mutations.csv')


def ts(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# Label loaders (identical to run_alpha_mixture_sweep.py)

def _load_base():
    meta   = pd.read_csv(OUT_DIR / 'patient_meta.csv')
    splits = json.load(open(OUT_DIR / 'pooled_random_splits.json'))
    train_set = set(splits[0]['cohort_a'])
    meta['held_out'] = ~meta['row_idx'].isin(train_set)
    ho = meta[meta['held_out']].copy().reset_index(drop=True)

    tmb_df = pd.read_csv(TMB_CSV).set_index('patient_id')
    muts   = pd.read_csv(MUTATION_CSV).set_index('patient_id')
    clin   = pd.read_csv(CLINICAL_CSV, low_memory=False)
    id_col = next(c for c in clin.columns if 'patient' in c.lower() and 'id' in c.lower())
    clin['pk'] = clin[id_col].apply(
        lambda x: '-'.join(str(x).split('-')[:3]) if isinstance(x, str) else x)
    ci = clin.drop_duplicates('pk').set_index('pk')

    ho['log_tmb'] = np.log1p(ho['patient_id'].map(tmb_df['tmb_per_mb']).astype(float))
    ho['tp53']    = ho['patient_id'].map(muts['TP53'])
    age_col = next((c for c in ci.columns if c.upper() == 'CLIN_AGE'), None)
    ho['age'] = pd.to_numeric(ho['patient_id'].map(ci[age_col]), errors='coerce') if age_col else np.nan
    ho['cancer_type'] = ho['patient_id'].map(ci['CLIN_PROJECT_ID'])
    return ho


def _get_pools(ho, lt):
    """Return (balanced, E, label_col) for a given label type."""
    if lt == 'tmb':
        h = ho[ho['log_tmb'].notna()].copy()
        med = h['log_tmb'].median()
        h['label'] = (h['log_tmb'] >= med).astype(np.int8)
        E_mask = h['log_tmb'] > h['log_tmb'].quantile(0.75)
    elif lt == 'tp53':
        h = ho[ho['tp53'].notna()].copy()
        h['label'] = h['tp53'].astype(np.int8)
        ct_prev   = h.groupby('cancer_type')['label'].mean()
        E_mask    = h['cancer_type'].isin(ct_prev[(ct_prev > 0.65) | (ct_prev < 0.15)].index)
    else:  # age
        h = ho[ho['age'].notna()].copy()
        med = h['age'].median()
        h['label'] = (h['age'] >= med).astype(np.int8)
        E_mask = h['age'] > h['age'].quantile(0.75)

    balanced = h[~E_mask].reset_index(drop=True)
    extreme  = h[E_mask].reset_index(drop=True)
    return balanced, extreme


# Split generation

# Seed offsets keep equal_N and vary_C independent across labels
_SEED_OFFSETS = {
    ('tmb',  'equal_N'): 10000,
    ('tmb',  'vary_C'):  11000,
    ('tp53', 'equal_N'): 22000, 
    ('tp53', 'vary_C'):  23000,
    ('age',  'equal_N'): 14000,
    ('age',  'vary_C'):  15000,
}


def make_equal_n_splits(balanced, extreme, lt, n, n_seeds=N_SEEDS):
    """A'=B=C=n. All three cohorts shrink together."""
    if n > len(extreme):
        return []   # not enough extreme patients for this N
    offset = _SEED_OFFSETS[(lt, 'equal_N')]
    splits = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + offset + n * 100)
        # A' and B from balanced pool (non-overlapping)
        if 2 * n > len(balanced):
            continue
        bp = rng.permutation(len(balanced))
        ap = balanced.iloc[bp[:n]]
        b  = balanced.iloc[bp[n:2*n]]
        # C from extreme pool
        ep = rng.permutation(len(extreme))
        c  = extreme.iloc[ep[:n]]
        splits.append({
            'label_type': lt, 'label_prefix': lt, 'design': 'equal_N', 'n': n, 'seed': seed,
            'idx_ap': ap['row_idx'].values, 'y_ap': ap['label'].values.astype(np.int8),
            'idx_b':  b['row_idx'].values,  'y_b':  b['label'].values.astype(np.int8),
            'idx_c':  c['row_idx'].values,
        })
    return splits


def make_vary_c_splits(balanced, extreme, lt, n_c, n_seeds=N_SEEDS):
    """A'/B fixed at full size; only C varies. Isolates D_quantile power."""
    if n_c > len(extreme):
        return []
    offset = _SEED_OFFSETS[(lt, 'vary_C')]
    splits = []
    for seed in range(n_seeds):
        rng_ab = np.random.default_rng(seed + offset)       # fixed A'/B across n_c
        rng_c  = np.random.default_rng(seed + offset + n_c * 100)  # varies with n_c
        bp = rng_ab.permutation(len(balanced))
        n_ap = len(bp) // 2
        ap = balanced.iloc[bp[:n_ap]]
        b  = balanced.iloc[bp[n_ap:]]
        ep = rng_c.permutation(len(extreme))
        c  = extreme.iloc[ep[:n_c]]
        splits.append({
            'label_type': lt, 'label_prefix': lt, 'design': 'vary_C', 'n': n_c, 'seed': seed,
            'idx_ap': ap['row_idx'].values, 'y_ap': ap['label'].values.astype(np.int8),
            'idx_b':  b['row_idx'].values,  'y_b':  b['label'].values.astype(np.int8),
            'idx_c':  c['row_idx'].values,
        })
    return splits


# Evaluation (identical pattern to other sweep scripts)

def _build_jobs(sp, model_names, repr_dir):
    lt = sp['label_type']; seed = sp['seed']; n = sp['n']; design = sp['design']
    jobs = []
    for mn in model_names:
        Zh, Zr, gamma = _load_cohorts(repr_dir, mn, sp['idx_ap'], sp['idx_b'], sp['idx_c'])
        is_fac = gamma is not None
        y_c = sp.get('y_c', np.ones(len(sp['idx_c']), dtype=np.int8))  # fallback: extreme pool always label=1
        jobs.append((
            seed, f'{lt}_power_{design}_N{n}', mn, is_fac, gamma,
            {'Ap': Zh['Ap'], 'B': Zh['B'], 'C': Zh['C']},
            {'Ap': Zr['Ap'], 'B': Zr['B'], 'C': Zr['C']},
            sp['y_ap'], sp['y_b'], y_c, lt, 0,
            _combo_seed(seed, f'{lt}_power_{design}', mn, lt, n),
        ))
    return jobs


def run_splits(split_list, model_names, repr_dir, res_dir, seed_only):
    for sp in split_list:
        lt = sp['label_type']; lp = sp.get('label_prefix', lt)
        design = sp['design']; n = sp['n']; seed = sp['seed']
        if seed_only is not None and seed != seed_only:
            continue
        prefix   = f"{lp}_power_{design}_N{n}_split"
        out_path = res_dir / f"{prefix}{seed}.json"
        if out_path.exists():
            continue
        jobs    = _build_jobs(sp, model_names, repr_dir)
        batches = Parallel(n_jobs=N_JOBS, backend='loky')(
            delayed(_safe_evaluate_one)(*j) for j in jobs)
        records = [r for b in batches for r in b]
        with open(out_path, 'w') as f:
            json.dump([_serialize(r) for r in records], f)


# Summary

def print_summary(res_dir):
    print("\n=== Power curve summary ===")
    for lt, lp in [('tmb','tmb'), ('tp53','tp53_uni'), ('age','age')]:
        for design in ('equal_N', 'vary_C'):
            print(f"\n{lt.upper()} — {design}")
            print(f"  {'N':>6}  {'S2%':>7}  {'S1%':>7}  {'n_records':>10}")
            for n in N_VALS:
                recs = []
                for f in glob.glob(str(res_dir / f'{lp}_power_{design}_N{n}_split*.json')):
                    recs.extend(json.load(open(f)))
                if not recs: continue
                s2 = sum(r['predicted_scenario']=='scenario_2' for r in recs)/len(recs)
                s1 = sum(r['predicted_scenario']=='scenario_1' for r in recs)/len(recs)
                print(f"  {n:>6}  {s2:>7.3f}  {s1:>7.3f}  {len(recs):>10}")


# Main

def main():
    parser = argparse.ArgumentParser(description='TCGA S2 detection power curve')
    parser.add_argument('--seed-only', type=int, default=None)
    parser.add_argument('--out-dir',   default=None)
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR  = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir  = OUT_DIR
    repr_dir = out_dir / 'representations'
    res_dir  = out_dir / 'results'
    res_dir.mkdir(parents=True, exist_ok=True)
    model_names = sorted(d.name for d in repr_dir.iterdir() if d.is_dir())

    ts(f"N_JOBS={N_JOBS}  N_SEEDS={N_SEEDS}  N_VALS={N_VALS}")
    ts(f"OUT_DIR={out_dir}")
    ts(f"Models: {model_names}")

    ho = _load_base()

    for lt in ('tmb', 'tp53', 'age'):
        ts(f"\n=== {lt.upper()} ===")
        balanced, extreme = _get_pools(ho, lt)
        lt_prefix = 'tp53_uni' if lt == 'tp53' else lt
        ts(f"  Balanced N={len(balanced)}, E N={len(extreme)}")

        # Build all splits for both designs across all N values
        all_splits = []
        for n in N_VALS:
            eq = make_equal_n_splits(balanced, extreme, lt, n)
            vc = make_vary_c_splits(balanced, extreme, lt, n)
            for sp in eq + vc: sp['label_prefix'] = lt_prefix
            if eq: ts(f"  equal_N N={n}: {len(eq)} splits (A'=B=C={n})")
            if vc: ts(f"  vary_C  N={n}: {len(vc)} splits (A'={len(balanced)//2}, B={len(balanced)//2}, C={n})")
            all_splits.extend(eq + vc)

        ts(f"  Total splits: {len(all_splits)}")
        run_splits(all_splits, model_names, repr_dir, res_dir, args.seed_only)

    ts("\n=== Done ===")
    print_summary(res_dir)


if __name__ == '__main__':
    main()
