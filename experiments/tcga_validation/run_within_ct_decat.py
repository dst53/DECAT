"""Within-cancer-type DECAT evaluation — unimodal FMs and multimodal representations.

Runs the full DECAT decision tree restricted to a single cancer type, detecting
confounding beyond cancer-type membership. Confounders are biological or clinical
(e.g. ER status, histological subtype, CDH1 mutation status)
and are never provided to DECAT — detection is label-free.

Design (vary-C):
  - A' (probe training): cohort_a patients of the selected type — large probe
  - Eval pool: held-out patients of the selected type, split 50/50 into B and c_pool
  - Extreme pool E: top-25% of c_pool by FM probe score (binary labels) or
    value-threshold (continuous labels: age, TMB, OS)
  - C: drawn from E at fraction α; B: random from eval pool (fixed, large)
  - α sweep: fraction of C from E ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
  - 50 splits per α, 5 FMs

Supported labels: age, tmb, msi, er, pr, os_cox, or any MC3 driver gene.

Post-hoc: TSS enrichment in E (checks whether extreme-probe patients cluster
by institution, without TSS labels being used during evaluation).

Output:
  <OUT_DIR>/<ct_tag>_<label>_fm_site_confounding_records.json
  <OUT_DIR>/<ct_tag>_<label>_fm_site_confounding_tss_enrichment.csv

Usage:
  # BRCA + TP53 (default):
  python experiments/tcga_validation/run_within_ct_decat.py

  # BRCA + ER status:
  python experiments/tcga_validation/run_within_ct_decat.py \\
      --cancer-type TCGA-BRCA --label er

  # BRCA + CDH1 mutation:
  python experiments/tcga_validation/run_within_ct_decat.py \\
      --cancer-type TCGA-BRCA --label CDH1
"""

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from evaluation.probes import fit_probe, predict_scores, cv_auroc, make_cv_splits
from sklearn.preprocessing import StandardScaler
from metrics.transfer import compute_transfer_ci, compute_transfer_null
from metrics.quantile import compute_dquantile_task

OUT_DIR     = Path(os.environ.get('DECAT_OUT_DIR', './data/tcga'))
N_PERM      = 200
N_BOOTSTRAP = 200
K_CV        = 5
N_SPLITS    = 50
ALPHAS      = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_JOBS      = max(1, os.cpu_count() - 2)

CLINICAL_CSV = os.environ.get('DECAT_CLINICAL_CSV', './data/tcga/CLINICAL.csv')

_FMB_DIR = os.environ.get('DECAT_DATA_DIR', './data/embeddings/pathology')
FM_PATHS = {
    'titan':        f'{_FMB_DIR}/titan/titan_20x.csv',
    'conch':        f'{_FMB_DIR}/conch/conch1_5_20x_MEAN.csv',
    'uni':          f'{_FMB_DIR}/uni/univ2_20x_MEAN.csv',
    'hoptimus0':    f'{_FMB_DIR}/hoptimus/hoptimus0_20x_MEAN.csv',
    'openmidnight': f'{_FMB_DIR}/openmidnight/openmidnight_20x_MEAN.csv',
}
FM_TYPE = {
    'titan': 'slide-level', 'conch': 'patch-MEAN', 'uni': 'patch-MEAN',
    'hoptimus0': 'patch-MEAN', 'openmidnight': 'patch-MEAN',
}


def ts(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_fm_embeddings(fm_name: str, patient_ids: np.ndarray) -> np.ndarray:
    """Load FM embeddings for given patient IDs. Returns (N, D) float32, NaN for missing."""
    path = FM_PATHS[fm_name]
    df = pd.read_csv(path, low_memory=False)
    df['patient_key'] = df['USUBJID'].apply(
        lambda x: '-'.join(str(x).replace('TCGA/', '').split('-')[:3]))
    df = df.drop_duplicates('patient_key').set_index('patient_key')
    ecols = [c for c in df.columns if c.startswith('E')]
    D = len(ecols)

    Z = np.full((len(patient_ids), D), np.nan, dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        if pid in df.index:
            Z[i] = df.loc[pid, ecols].values.astype(np.float32)

    n_valid = (~np.isnan(Z[:, 0])).sum()
    ts(f"  {fm_name}: {n_valid}/{len(patient_ids)} patients with embeddings, dim={D}")
    return Z


def load_repr_embeddings(model_name: str, out_dir: Path, n_patients: int) -> np.ndarray:
    """Load multimodal H&E representations (Zh) from trained model.

    Representations are aligned to the full patient_meta.csv ordering,
    so no per-patient lookup is needed — the row index IS the patient index.

    Parameters
    ----------
    model_name : str
        Model directory name, e.g. 'clip_K50', 'cca_K50', 'jive_K50'.
    out_dir : Path
        DECAT output directory containing representations/.
    n_patients : int
        Total number of patients (for shape validation).

    Returns
    -------
    Z : (n_patients, d) float32 — no NaN rows (all patients have representations)
    """
    path = out_dir / 'representations' / model_name / 'full_latents.npz'
    data = np.load(path)
    Z = data['Zh'].astype(np.float32)
    assert Z.shape[0] == n_patients, f"Shape mismatch: {Z.shape[0]} vs {n_patients}"
    ts(f"  {model_name}: {Z.shape[0]} patients, dim={Z.shape[1]} (H&E repr)")
    return Z


def _verdict(signal_present, ptf_fires, dq_fires):
    if not signal_present:
        return 'S3'
    if ptf_fires and dq_fires:
        return 'S2'
    if ptf_fires and not dq_fires:
        return 'Transferable'
    return 'Indeterminate'


def evaluate_split(
    fm_name, split_id, alpha, Z_all, y_all, train_indices, eval_indices, rng_seed,
    y_raw=None, n_pca=None,
):
    """Run one (FM, split, alpha) evaluation.

    Vary-C design: A' = all cohort_a BRCA patients (~571, good probe quality).
    Eval pool is split 50/50 into B and c_pool. C is drawn from the extreme
    pool E (top-25% of c_pool by probe score).
    """
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    try:
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(rng_seed)

        # A': cohort_a BRCA patients with embeddings + label
        ap_idx = np.array([i for i in train_indices
                           if not np.isnan(Z_all[i, 0]) and y_all[i] >= 0])
        if len(ap_idx) < 50:
            return None

        # Eval pool: split 50/50 into B and c_pool
        valid_eval = np.array([i for i in eval_indices
                               if not np.isnan(Z_all[i, 0]) and y_all[i] >= 0])
        if len(valid_eval) < 40:
            return None
        rng_split = np.random.default_rng(rng.integers(1 << 63))
        perm = rng_split.permutation(valid_eval)
        half = len(perm) // 2
        b_idx = perm[:half]
        c_pool = perm[half:]

        # Standardize (and optionally PCA-compress) using A' statistics
        Z_ap = Z_all[ap_idx]
        y_ap = y_all[ap_idx]
        scaler = StandardScaler().fit(Z_ap)
        Z_ap_std = scaler.transform(Z_ap).astype(np.float32)

        if n_pca is not None and n_pca < Z_ap_std.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_pca, random_state=42).fit(Z_ap_std)
            Z_ap_std = pca.transform(Z_ap_std).astype(np.float32)
        else:
            pca = None

        def _preprocess(Z):
            out = scaler.transform(Z).astype(np.float32)
            return pca.transform(out).astype(np.float32) if pca is not None else out

        # Define extreme pool E
        if y_raw is not None and not isinstance(y_raw, dict):
            # Value-threshold E: top-25% of c_pool by actual label value.
            # Consistent with manuscript design for continuous labels (TMB, Age).
            # (y_raw is a dict for os_cox survival data — falls through to probe-score E)
            cpool_vals = y_raw[c_pool]
            valid_cpool = ~np.isnan(cpool_vals)
            p75 = np.nanpercentile(cpool_vals, 75)
            e_mask = valid_cpool & (cpool_vals >= p75)
            e_indices = c_pool[e_mask]
            non_e_indices = c_pool[~e_mask]
        else:
            # Probe-score E: top-25% by FM probe prediction.
            # Used for binary labels (genes) where no continuous value is available.
            _, clf = fit_probe(Z_ap_std, y_ap, probe_backend='logistic',
                               C=0.1, solver='liblinear', max_iter=500,
                               standardize=False)
            Z_cpool_std = _preprocess(Z_all[c_pool])
            scores_cpool = predict_scores(None, clf, Z_cpool_std)
            n_e = max(20, len(c_pool) // 4)
            e_order = np.argsort(scores_cpool)[::-1]
            e_indices = c_pool[e_order[:n_e]]
            non_e_indices = c_pool[e_order[n_e:]]

        # Also train probe on A' (needed regardless of E type — for signal gate + metrics)
        _, clf = fit_probe(Z_ap_std, y_ap, probe_backend='logistic',
                           C=0.1, solver='liblinear', max_iter=500,
                           standardize=False)

        # Build C via alpha mixture: fraction α from E, rest from non-E
        n_c = min(len(e_indices), len(non_e_indices), len(c_pool) // 2)
        n_from_e = int(alpha * n_c)
        n_from_rest = n_c - n_from_e

        if n_from_e > len(e_indices) or n_from_rest > len(non_e_indices):
            return None

        rng_c = np.random.default_rng(rng.integers(1 << 63))
        c_idx = np.concatenate([
            rng_c.choice(e_indices, size=n_from_e, replace=False),
            rng_c.choice(non_e_indices, size=n_from_rest, replace=False),
        ])

        # Extract cohort arrays
        Z_b_std = _preprocess(Z_all[b_idx])
        Z_c_std = _preprocess(Z_all[c_idx])
        y_b = y_all[b_idx]
        y_c = y_all[c_idx]

        # Raw (unstandardized) for D_quantile
        Z_ap_raw = Z_all[ap_idx].astype(np.float32)
        Z_b_raw = Z_all[b_idx].astype(np.float32)
        Z_c_raw = Z_all[c_idx].astype(np.float32)

        # Stage II: signal gate
        splits_cv = make_cv_splits(len(y_ap), y_ap, n_folds=K_CV, rng=rng)
        observed = cv_auroc(Z_ap_std, y_ap, splits=splits_cv)
        null_seeds = rng.integers(1 << 63, size=N_PERM)
        null_vals = np.array([
            cv_auroc(Z_ap_std, np.random.default_rng(s).permutation(y_ap), splits=splits_cv)
            for s in null_seeds
        ])
        p_signal = (1 + np.sum(null_vals >= observed)) / (N_PERM + 1)
        signal_present = bool(p_signal < 0.05)

        # Stage IV: P_transfer + D_quantile
        t_ci = compute_transfer_ci(Z_ap_std, y_ap, Z_b_std, y_b, Z_c_std,
                                   n_bootstrap=N_BOOTSTRAP,
                                   rng=np.random.default_rng(rng.integers(1 << 63)))
        t_null = compute_transfer_null(Z_ap_std, y_ap, Z_b_std, y_b, Z_c_std,
                                       n_permutations=N_PERM,
                                       rng=np.random.default_rng(rng.integers(1 << 63)))
        # Build survival_data dict for Cox probe if y_raw carries OS data
        surv_data = None
        if isinstance(y_raw, dict) and 'times' in y_raw:
            surv_data = {
                'times':  y_raw['times'][ap_idx],
                'events': y_raw['events'][ap_idx],
            }
        dq = compute_dquantile_task(Z_ap_raw, Z_b_raw, Z_c_raw, y_ap,
                                    n_permutations=N_PERM,
                                    rng=np.random.default_rng(rng.integers(1 << 63)),
                                    survival_data=surv_data)

        ptf_fires = bool(t_ci['ci_lower'] > t_null['null_upper'])
        dq_fires = bool(dq['dquantile_task'] >= dq['dquantile_task_null_upper'])

        return {
            'fm': fm_name, 'fm_type': FM_TYPE.get(fm_name, 'multimodal-repr'),
            'split_id': split_id, 'alpha': alpha,
            'label': None, 'cancer_type': None,  # filled by caller
            'n_ap': len(ap_idx), 'n_b': len(b_idx), 'n_c': len(c_idx),
            'n_e': len(e_indices), 'design': 'vary_c',
            'auroc': round(float(observed), 4),
            'signal_present': signal_present,
            'p_transfer': float(t_ci['p_transfer']),
            'transfer_ci_lower': float(t_ci['ci_lower']),
            'transfer_null_upper': float(t_null['null_upper']),
            'dquantile_task': float(dq['dquantile_task']),
            'dquantile_null_upper': float(dq['dquantile_task_null_upper']),
            'ptf_fires': ptf_fires, 'dq_fires': dq_fires,
            'verdict': _verdict(signal_present, ptf_fires, dq_fires),
            'e_patient_indices': e_indices.tolist(),
        }
    except Exception as e:
        import traceback
        ts(f"  ERROR {fm_name}/split{split_id}/a{alpha}: {e}")
        return None


def _load_label(label: str, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Load and binarize a label for all patients.

    Returns
    -------
    y_binary : (N,) float — binarized at median, -1 for missing
    y_raw    : (N,) float or None — raw continuous values (for value-threshold E),
               None for binary gene labels
    """
    y = np.full(len(meta), -1.0)
    y_raw = None
    pids = meta['patient_id'].values

    clinical = pd.read_csv(CLINICAL_CSV, low_memory=False)
    clinical['pid12'] = clinical['patient_id'].str[:12]

    if label == 'age':
        age_map = dict(zip(clinical['pid12'], pd.to_numeric(clinical['CLIN_AGE'], errors='coerce')))
        vals = np.array([age_map.get(p, np.nan) for p in pids])
        valid = ~np.isnan(vals)
        median = np.nanmedian(vals)
        y[valid] = (vals[valid] > median).astype(float)
        y_raw = vals  # raw continuous values for value-threshold E
        ts(f"Age: {valid.sum()} patients, median={median:.1f}")

    elif label == 'tmb':
        tmb = pd.read_csv(os.environ.get('DECAT_TMB_CSV',
                          './data/tcga/tcga_tmb.csv'))
        tmb['pid12'] = tmb['patient_id'].str[:12]
        tmb_map = dict(zip(tmb['pid12'], tmb['tmb_per_mb']))
        vals = np.array([tmb_map.get(p, np.nan) for p in pids])
        valid = ~np.isnan(vals)
        median = np.nanmedian(vals)
        y[valid] = (vals[valid] > median).astype(float)
        y_raw = vals  # raw continuous values for value-threshold E
        ts(f"TMB: {valid.sum()} patients, median={median:.2f}")

    elif label == 'os_cox':
        # Overall survival — binarized at median for signal gate + E selection,
        # raw times/events stored in y_raw as a dict for Cox D_quantile probe.
        clin_os = pd.read_csv(CLINICAL_CSV, low_memory=False)
        clin_os['pid12'] = clin_os['patient_id'].str[:12]
        os_months_map = dict(zip(clin_os['pid12'], pd.to_numeric(clin_os['CLIN_OS_MONTHS'], errors='coerce')))
        os_event_map  = dict(zip(clin_os['pid12'], (clin_os['CLIN_OS_STATUS'] == '1:DECEASED').astype(float)))
        times  = np.array([float(os_months_map.get(p, np.nan)) for p in pids])
        events = np.array([float(os_event_map.get(p, np.nan)) for p in pids])
        valid  = (times > 0) & ~np.isnan(times)
        median = np.nanmedian(times[valid])
        y[valid] = (times[valid] > median).astype(float)
        y_raw = {'times': times, 'events': events}
        ts(f"OS Cox: {valid.sum()} patients, events={(events[valid]==1).sum():.0f}, "
           f"median_OS={median:.1f}mo")

    elif label == 'msi':
        msi = pd.read_csv('./data/tcga/tcga_msi_sensor_scores.csv')
        msi['pid12'] = msi['patient_id'].str[:12]
        msi_map = dict(zip(msi['pid12'], (msi['msi_status'] == 'MSI-H').astype(float)))
        n_valid = 0
        for i, pid in enumerate(pids):
            if pid in msi_map:
                y[i] = msi_map[pid]
                n_valid += 1
        ts(f"MSI: {n_valid} patients, MSI-H prevalence={y[y >= 0].mean():.2f}")

    elif label in ('er', 'pr'):
        # ER/PR status from IHC (BRCA-specific)
        er_pr = pd.read_csv('./data/tcga/tcga_brca_er_pr_status.csv')
        er_pr['pid12'] = er_pr['patient_id'].str[:12]
        col = 'er_status_ihc' if label == 'er' else 'pr_status_ihc'
        status_map = dict(zip(er_pr['pid12'], er_pr[col]))
        n_valid = 0
        for i, pid in enumerate(pids):
            status = status_map.get(pid)
            if status == 'Positive':
                y[i] = 1.0
                n_valid += 1
            elif status == 'Negative':
                y[i] = 0.0
                n_valid += 1
        ts(f"{label.upper()}: {n_valid} patients, "
           f"prevalence={y[y >= 0].mean():.2f}")

    else:
        # Assume driver gene mutation (binary 0/1 in MC3 matrix)
        muts = pd.read_csv(os.environ.get('DECAT_MUTATION_CSV',
                           './data/tcga/tcga_driver_mutations.csv'))
        muts['pid12'] = muts['patient_id'].str[:12]
        # Gene names are uppercase in the mutation matrix
        label = label.upper()
        if label not in muts.columns:
            raise ValueError(f"Label '{label}' not found. Available: "
                             f"{[c for c in muts.columns if c not in ('patient_id','pid12')]}")
        muts = muts.set_index('pid12')
        for i, pid in enumerate(pids):
            if pid in muts.index:
                v = muts.loc[pid, label]
                if v >= 0:
                    y[i] = float(v)
        ts(f"{label}: {(y >= 0).sum()} patients, "
           f"prevalence={y[y >= 0].mean():.2f}")

    return y, y_raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Within-cancer-type site confounding detection")
    parser.add_argument('--cancer-type', default='TCGA-BRCA',
                        help='TCGA cancer type code (default: TCGA-BRCA). '
                             'Use TCGA-LUSC for LUSC, etc.')
    parser.add_argument('--label', default='age',
                        help='Label to predict: age, tmb, tp53, or any MC3 gene '
                             '(default: age)')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--n-a-prime', type=int, default=None,
                        help='Override A-prime size: randomly partition all available '
                             'patients into N_A_PRIME (A-prime) + N_B (B) + remainder '
                             '(c_pool). Enables larger C at cost of smaller probe '
                             'training sets. Best for labels with strong morphological '
                             'signal (e.g. MSI). Default: use cohort_a as A-prime.')
    parser.add_argument('--n-b', type=int, default=None,
                        help='Override B size when --n-a-prime is set '
                             '(default: same as --n-a-prime).')
    parser.add_argument('--pca-components', type=int, default=None,
                        help='Apply PCA to FM embeddings before probe training '
                             '(e.g. 50). Helps when N_A_prime is small relative '
                             'to embedding dimensionality. Skipped for multimodal '
                             'repr (already 50-dim).')
    parser.add_argument('--repr-model', default=None,
                        help='Use multimodal H&E representations instead of FM '
                             'embeddings, e.g. clip_K50, cca_K50, jive_K50. '
                             'Loads Zh from representations/<model>/full_latents.npz.')
    args = parser.parse_args()

    out_dir   = Path(args.out_dir) if args.out_dir else OUT_DIR
    ct        = args.cancer_type
    label     = args.label.lower()
    ct_tag    = ct.replace('TCGA-', '').lower()
    repr_tag  = args.repr_model or 'fm'
    pca_tag   = f'_pca{args.pca_components}' if args.pca_components else ''
    out_stem  = f'{ct_tag}_{label}_{repr_tag}{pca_tag}_site_confounding'
    n_pca     = args.pca_components if not args.repr_model else None  # skip PCA for multimodal repr
    n_ap_override = args.n_a_prime
    n_b_override  = args.n_b if args.n_b else args.n_a_prime
    if n_ap_override:
        n_ap_tag  = f'_ap{n_ap_override}b{n_b_override}'
        out_stem  = f'{ct_tag}_{label}_{repr_tag}{pca_tag}{n_ap_tag}_site_confounding'

    ts(f"=== Within-{ct_tag.upper()} site confounding  label={label}  OUT_DIR={out_dir} ===")

    # Load patient metadata
    meta = pd.read_csv(out_dir / 'patient_meta.csv')
    ts(f"Total patients: {len(meta)}")

    # Filter to selected cancer type
    ct_mask = meta['cancer_type'] == ct
    ct_indices = np.where(ct_mask.values)[0]
    ts(f"{ct} patients: {len(ct_indices)}")
    if len(ct_indices) < 100:
        raise ValueError(f"Too few patients for {ct}: {len(ct_indices)}")

    # Load label (returns binarized array + optional raw continuous values)
    y_all, y_raw = _load_label(label, meta)

    if n_ap_override:
        # Rebalanced design: randomly partition all available patients.
        # A' and B are set to fixed sizes; remainder becomes c_pool → larger E and C.
        # Use a fixed seed so splits are reproducible across FMs.
        all_valid = [i for i in ct_indices if y_all[i] >= 0]
        rng_split = np.random.default_rng(999)
        perm = rng_split.permutation(all_valid)
        ct_train = list(perm[:n_ap_override])
        ct_eval  = list(perm[n_ap_override:])   # B + c_pool combined
        ts(f"{ct_tag.upper()} rebalanced: A'={len(ct_train)} (N={n_ap_override}), "
           f"eval pool (B+C): {len(ct_eval)} → E≈{len(ct_eval)//4} @ top-25%")
    else:
        # Default vary-C design: A' = cohort_a patients.
        pooled_splits = json.load(open(out_dir / 'pooled_random_splits.json'))
        train_set = set(pooled_splits[0]['cohort_a'])
        ct_train = [i for i in ct_indices if i in train_set and y_all[i] >= 0]
        ct_eval  = [i for i in ct_indices if i not in train_set and y_all[i] >= 0]
        ts(f"{ct_tag.upper()} A' (train, with label): {len(ct_train)}, "
           f"eval pool (B+C): {len(ct_eval)}")

    # Determine which representations to iterate over
    if args.repr_model:
        model_list = [args.repr_model]
    else:
        model_list = list(FM_PATHS.keys())

    # Run evaluation for each model/FM
    all_records = []
    for model_name in model_list:
        ts(f"\nLoading {model_name}...")
        if args.repr_model:
            Z_all = load_repr_embeddings(model_name, out_dir, len(meta))
            # Multimodal repr: all patients have representations (no NaN)
            train_valid = [i for i in ct_train]
            eval_valid  = [i for i in ct_eval]
        else:
            Z_all = load_fm_embeddings(model_name, meta['patient_id'].values)
            train_valid = [i for i in ct_train if not np.isnan(Z_all[i, 0])]
            eval_valid  = [i for i in ct_eval  if not np.isnan(Z_all[i, 0])]

        ts(f"  A' (train) with embeddings: {len(train_valid)}, "
           f"eval pool: {len(eval_valid)}")

        # Build jobs
        jobs = []
        for alpha in ALPHAS:
            for split_id in range(N_SPLITS):
                seed = split_id * 1000 + int(alpha * 100)
                jobs.append((model_name, split_id, alpha, Z_all, y_all,
                            train_valid, eval_valid, seed, y_raw, n_pca))

        ts(f"  Running {len(jobs)} evaluations (N_JOBS={N_JOBS})...")
        results = Parallel(n_jobs=N_JOBS, backend='loky')(
            delayed(evaluate_split)(*job) for job in jobs
        )
        results = [r for r in results if r is not None]
        for r in results:
            r['label'] = label
            r['cancer_type'] = ct
        all_records.extend(results)
        ts(f"  {model_name}: {len(results)} records")

    # Save results
    out_path = out_dir / f'{out_stem}_records.json'
    with open(out_path, 'w') as f:
        json.dump(all_records, f)
    ts(f"\nSaved {len(all_records)} records to {out_path}")

    # Post-hoc TSS enrichment: check if E concentrates specific institutions
    ts("\n=== Post-hoc TSS enrichment analysis ===")
    meta_tss = meta.copy()
    meta_tss['tss'] = meta_tss['patient_id'].str[5:7]

    enrichment_rows = []
    for fm_name in model_list:
        fm_records = [r for r in all_records if r['fm'] == fm_name and r['alpha'] == 1.0]
        if not fm_records:
            continue
        # Aggregate E indices across splits
        e_tss_counts = {}
        eval_tss_counts = {}
        for r in fm_records:
            for idx in r['e_patient_indices']:
                tss = meta_tss.iloc[idx]['tss']
                e_tss_counts[tss] = e_tss_counts.get(tss, 0) + 1
        for idx in ct_eval:
            tss = meta_tss.iloc[idx]['tss']
            eval_tss_counts[tss] = eval_tss_counts.get(tss, 0) + 1

        # Top-5 TSS in E
        top_e = sorted(e_tss_counts.items(), key=lambda x: -x[1])[:5]
        total_e = sum(e_tss_counts.values())
        total_eval = sum(eval_tss_counts.values())
        ts(f"  {fm_name}: top-5 TSS in E = {[(t, f'{c/total_e:.1%}') for t, c in top_e]}")

        for tss, e_count in e_tss_counts.items():
            eval_count = eval_tss_counts.get(tss, 0)
            enrichment_rows.append({
                'fm': fm_name, 'tss': tss,
                'e_count': e_count, 'e_frac': e_count / total_e,
                'eval_count': eval_count, 'eval_frac': eval_count / total_eval,
                'enrichment': (e_count / total_e) / (eval_count / total_eval + 1e-8),
            })

    enrichment_df = pd.DataFrame(enrichment_rows)
    enrich_path = out_dir / f'{out_stem}_tss_enrichment.csv'
    enrichment_df.to_csv(enrich_path, index=False)
    ts(f"Saved TSS enrichment to {enrich_path}")

    # Summary
    ts("\n=== Summary ===")
    for fm_name in model_list:
        for alpha in ALPHAS:
            sub = [r for r in all_records if r['fm'] == fm_name and r['alpha'] == alpha]
            if not sub:
                continue
            s2_rate = sum(1 for r in sub if r['verdict'] == 'S2') / len(sub)
            trans_rate = sum(1 for r in sub if r['verdict'] == 'Transferable') / len(sub)
            indet_rate = sum(1 for r in sub if r['verdict'] == 'Indeterminate') / len(sub)
            if alpha in (0.0, 0.4, 1.0):
                ts(f"  {fm_name} α={alpha:.1f}: S2={s2_rate:.0%} Trans={trans_rate:.0%} "
                   f"Indet={indet_rate:.0%} (N={len(sub)})")


if __name__ == '__main__':
    main()
