"""Alpha-mixture extreme-C sweep — S1→S2 transition curve.

Design
------
Held-out patients are split 3 ways per seed:

  A' (~33%)    probe training + P_transfer reference
  B  (~33%)    P_transfer evaluation
  C_pool (~34%) candidate deployment cohort pool

At each alpha ∈ {0.0, 0.1, …, 1.0}, C is drawn from C_pool as a mixture:

  C = alpha × N_C patients from E ∩ C_pool   (label-extreme)
    + (1-alpha) × N_C patients from remaining C_pool  (random)

where N_C = min(|E ∩ C_pool|, 300) is fixed across all alpha so that
statistical power (D_quantile N_C) is constant and any change in
detection rate is due to C composition, not cohort size.

x-axis interpretation: "fraction of deployment cohort drawn from the
label-extreme pool E."

  alpha=0 → C random from C_pool → same ~25% E composition as A'/B → S1
  alpha=1 → C = pure E ∩ C_pool  → maximally label-extreme → S2

Cohort sizes (approximate, TITAN v3, TMB label example)
-------------------------------------------------------
  A' : ~1,465 patients  (33% of held-out with valid label, N≈4,395)
  B  : ~1,465 patients
  C  :    300 patients  (N_C fixed = min(|E ∩ C_pool|, 300) ≈ 300)

A'/B are intentionally larger than C. This is by design:
  - P_transfer needs large A'/B for stable probe training and ranking
  - D_quantile needs C large enough for reliable Wasserstein estimation;
    N_C=300 is sufficient (permutation null uses same N_C, so power is
    self-consistent) and reflects a realistic deployment cohort size
  - The asymmetry does not bias the metrics: P_transfer ignores C,
    D_quantile's null shuffles B/C labels with fixed probe direction

Extreme pool E (per label)
--------------------------
  TMB:  log(TMB+1) > p75  — enriched for SKCM, UCEC, LUSC (η²=0.44)
  Age:  age > p75          — enriched for PRAD, KIRC, BRCA  (η²=0.27)

50 seeds per alpha.  Results write to:
  <OUT_DIR>/results/{label}_v3_mix_a{pct}_split{seed}.json
"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parents[2]))
from experiments.tcga_validation.run_alpha_mixture_sweep_probe_e import (
    make_mixture_splits,
)
from experiments.tcga_validation.run_decat import (
    _safe_evaluate_one, _load_cohorts, _combo_seed, _serialize,
)

_DEFAULT_OUT_DIR = Path(os.environ.get('DECAT_OUT_DIR', './data/tcga'))
N_SEEDS  = 50
ALPHAS   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_JOBS   = max(1, os.cpu_count() - 2)

def ts(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# Module-level mutable references — set in main() or at module load if run directly
OUT_DIR     = _DEFAULT_OUT_DIR
repr_dir    = OUT_DIR / 'representations'
model_names: list = []
ho_all      = None
tmb_df      = muts = clin = None


def _load_globals(out_dir: Path) -> None:
    """Load all data needed by label loaders and build_jobs. Called from main()."""
    global OUT_DIR, repr_dir, model_names, ho_all, tmb_df, muts, clin
    OUT_DIR  = out_dir
    repr_dir = out_dir / 'representations'
    model_names = sorted(d.name for d in repr_dir.iterdir() if d.is_dir())
    ts(f"N_JOBS={N_JOBS}  models={model_names}")

    meta   = pd.read_csv(out_dir / 'patient_meta.csv')
    splits = json.load(open(out_dir / 'pooled_random_splits.json'))
    train_set = set(splits[0]['cohort_a'])
    meta['held_out'] = ~meta['row_idx'].isin(train_set)
    ho_all = meta[meta['held_out']].copy().reset_index(drop=True)

    tmb_df  = pd.read_csv(os.environ.get('DECAT_TMB_CSV', './data/tcga/tcga_tmb.csv')).set_index('patient_id')
    muts    = pd.read_csv(os.environ.get('DECAT_MUTATION_CSV', './data/tcga/tcga_driver_mutations.csv')).set_index('patient_id')
    clin_raw = pd.read_csv(os.environ.get('DECAT_CLINICAL_CSV', './data/tcga/CLINICAL.csv'), low_memory=False)
    id_col   = next(c for c in clin_raw.columns if 'patient' in c.lower() and 'id' in c.lower())
    clin_raw['pk'] = clin_raw[id_col].apply(
        lambda x: '-'.join(str(x).split('-')[:3]) if isinstance(x, str) else x)
    clin = clin_raw.drop_duplicates('pk').set_index('pk')


# Label loaders
def load_tmb():
    ho = ho_all.copy()
    ho['log_tmb'] = np.log1p(ho['patient_id'].map(tmb_df['tmb_per_mb']).astype(float))
    ho = ho[ho['log_tmb'].notna()].reset_index(drop=True)
    med = ho['log_tmb'].median()
    ho['label'] = (ho['log_tmb'] >= med).astype(np.int8)
    extreme_mask = ho['log_tmb'] > ho['log_tmb'].quantile(0.75)
    ts(f"  TMB: N={len(ho)}, |E|={extreme_mask.sum()}")
    return ho, extreme_mask

def load_age():
    age_col = next((c for c in clin.columns if c.upper() == 'CLIN_AGE'), None)
    ho = ho_all.copy()
    ho['age'] = pd.to_numeric(ho['patient_id'].map(clin[age_col]), errors='coerce') if age_col else np.nan
    ho = ho[ho['age'].notna()].reset_index(drop=True)
    med = ho['age'].median()
    ho['label'] = (ho['age'] >= med).astype(np.int8)
    extreme_mask = ho['age'] > ho['age'].quantile(0.75)
    ts(f"  Age: N={len(ho)}, |E|={extreme_mask.sum()}")
    return ho, extreme_mask


def build_jobs(sp, lt):
    seed = sp['seed']; alpha = sp['alpha']
    jobs = []
    for mn in model_names:
        Zh, Zr, gamma = _load_cohorts(repr_dir, mn, sp['idx_ap'], sp['idx_b'], sp['idx_c'])
        is_fac = gamma is not None
        jobs.append((
            seed, f'{lt}_v2_mix_a{alpha:.2f}', mn, is_fac, gamma,
            {'Ap': Zh['Ap'], 'B': Zh['B'], 'C': Zh['C']},
            {'Ap': Zr['Ap'], 'B': Zr['B'], 'C': Zr['C']},
            sp['y_ap'], sp['y_b'], sp['y_c'], lt, 0,
            _combo_seed(seed, f'{lt}_v2_mix', mn, lt, int(alpha * 100)),
        ))
    return jobs


def run_label(label_type, ho, extreme_mask):
    res_dir = OUT_DIR / 'results'
    res_dir.mkdir(parents=True, exist_ok=True)
    splits  = make_mixture_splits(ho, extreme_mask, label_type)
    ts(f"\n{label_type.upper()} v2: {len(splits)} splits ({N_SEEDS} seeds × {len(ALPHAS)} alphas)")

    for sp in splits:
        alpha    = sp['alpha']
        seed     = sp['seed']
        out_path = res_dir / f"{label_type}_v3_mix_a{int(alpha*100):03d}_split{seed}.json"
        if out_path.exists():
            continue
        jobs    = build_jobs(sp, label_type)
        batches = Parallel(n_jobs=N_JOBS, backend='loky')(
            delayed(_safe_evaluate_one)(*j) for j in jobs)
        records = [r for b in batches for r in b]
        with open(out_path, 'w') as f:
            json.dump([_serialize(r) for r in records], f)

    # Summary
    ts(f"\n  {label_type.upper()} v2 — S2 / indeterminate / S3 rate by alpha:")
    print(f"  {'alpha':>6}  {'S1':>6}  {'S2':>6}  {'Ind':>6}  {'S3':>6}")
    for alpha in ALPHAS:
        recs = []
        for f in glob.glob(str(res_dir / f"{label_type}_v3_mix_a{int(alpha*100):03d}_split*.json")):
            recs.extend(json.load(open(f)))
        if not recs: continue
        n = len(recs)
        s1  = sum(r['predicted_scenario']=='scenario_1'   for r in recs)/n
        s2  = sum(r['predicted_scenario']=='scenario_2'   for r in recs)/n
        ind = sum(r['predicted_scenario']=='indeterminate' for r in recs)/n
        s3  = sum(r['predicted_scenario']=='scenario_3'   for r in recs)/n
        print(f"  {alpha:>6.1f}  {s1:>6.2f}  {s2:>6.2f}  {ind:>6.2f}  {s3:>6.2f}")


# Main

def main():
    import argparse as _ap
    parser = _ap.ArgumentParser(description='Alpha-mixture sweep')
    parser.add_argument('--out-dir',   default=None)
    parser.add_argument('--seed-only', type=int, default=None,
                        help='Run only this seed for quick sanity check')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _DEFAULT_OUT_DIR
    _load_globals(out_dir)

    ts("=== Alpha-mixture sweep v2 (corrected 3-way split) ===")
    ts(f"OUT_DIR={out_dir}")

    for lt, loader in [('tmb', load_tmb), ('age', load_age)]:
        ts(f"Loading {lt.upper()} ...")
        ho, mask = loader()
        run_label(lt, ho, mask)

    ts("=== Done ===")


if __name__ == '__main__':
    main()
