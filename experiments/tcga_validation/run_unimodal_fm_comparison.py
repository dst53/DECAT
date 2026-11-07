"""Unimodal H&E FM comparison — DECAT Stages II + IV only.

Evaluates five pathology foundation models (TITAN, CONCH, UNI, hoptimus0,
openmidnight) using DECAT's confound-detection metrics without a paired RNA
modality. Stages I (A_norm) and III (Δ_shared) require two modalities and are
therefore skipped. This produces:

  Stage II  — signal gate: is the label predictable from H&E at all?
  Stage IV  — P_transfer + D_quantile^task: does the prediction transfer
              across cohorts and is the ordering stable?

Verdict mapping (unimodal subset of DECAT):
  S3               — signal gate fails (no predictable signal)
  S2               — P_transfer fires AND D_quantile fires (confounded)
  Transferable     — P_transfer fires, D_quantile stable (S1/S4-like;
                     RNA required to distinguish shared from modality-specific)
  Indeterminate    — D_quantile fires but P_transfer fails, or neither fires
                     despite signal gate passing

Two split types:
  pooled_random    — balanced pan-cancer cohorts → expect Transferable (S1-like)
  extreme_c        — C is label-extreme → expect S2 (confound revealed)

Default labels: TMB (η²=0.44), TP53 (η²=0.34), Age (η²=0.27); KRAS (η²=0.32) via --labels

Output:
  <OUT_DIR>/unimodal_fm_comparison_records.json   — per-split records
  <OUT_DIR>/unimodal_fm_comparison_summary.csv    — summary table

Usage:
  python experiments/tcga_validation/run_unimodal_fm_comparison.py
  python experiments/tcga_validation/run_unimodal_fm_comparison.py --out-dir <dir>
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

# Default output directory — override with --out-dir
OUT_DIR      = Path(os.environ.get('DECAT_OUT_DIR', './data/tcga'))
N_PERM       = 200
N_BOOTSTRAP  = 200
K_CV         = 5
N_JOBS       = max(1, os.cpu_count() - 2)

# External data paths for new labels
CLINICAL_CSV = os.environ.get('DECAT_CLINICAL_CSV', './data/tcga/CLINICAL.csv')
MUTATION_CSV = os.environ.get('DECAT_MUTATION_CSV', './data/tcga/tcga_driver_mutations.csv')
TMB_CSV      = os.environ.get('DECAT_TMB_CSV', './data/tcga/tcga_tmb.csv')

DEFAULT_LABELS = ('tmb', 'tp53', 'age')

_FMB_DIR = os.environ.get('DECAT_DATA_DIR', './data/embeddings/pathology')
FM_PATHS = {
    'titan':        f'{_FMB_DIR}/titan/titan_20x.csv',
    'conch':        f'{_FMB_DIR}/conch/conch1_5_20x_MEAN.csv',
    'uni':          f'{_FMB_DIR}/uni/univ2_20x_MEAN.csv',
    'hoptimus0':    f'{_FMB_DIR}/hoptimus/hoptimus0_20x_MEAN.csv',
    'openmidnight': f'{_FMB_DIR}/openmidnight/openmidnight_20x_MEAN.csv',
}

# task_idx in task_labels.npz (pooled S2B labels)
LABEL_TASK_IDX = {'tmb': 52, 'tp53': 51, 'age': 50}
LABEL_ETA2     = {'tmb': 0.44, 'tp53': 0.34, 'age': 0.27}

FM_TYPE = {
    'titan':        'slide-level',
    'conch':        'patch-MEAN',
    'uni':          'patch-MEAN',
    'hoptimus0':    'patch-MEAN',
    'openmidnight': 'patch-MEAN',
}

def compute_eta2(y: np.ndarray, cancer_types: np.ndarray) -> float:
    """Fraction of label variance explained by cancer type (η²)."""
    valid = y >= 0
    yv, ctv = y[valid].astype(float), cancer_types[valid]
    grand = yv.mean()
    ss_b = sum(np.sum(ctv == ct) * (yv[ctv == ct].mean() - grand) ** 2
               for ct in np.unique(ctv))
    ss_t = ((yv - grand) ** 2).sum()
    return float(ss_b / ss_t) if ss_t > 0 else 0.0


def load_external_labels(meta: pd.DataFrame) -> dict:
    """Load KRAS mutation status from MC3.

    Returns a dict mapping label name → np.int8 array of shape (N,), -1=missing.
    """
    n = len(meta)
    pids = meta['patient_id'].values

    muts = pd.read_csv(MUTATION_CSV).set_index('patient_id')
    kras = np.full(n, -1, dtype=np.int8)
    for i, pid in enumerate(pids):
        if pid in muts.index:
            v = muts.loc[pid, 'KRAS']
            if v >= 0:
                kras[i] = int(v)

    return {'kras': kras}


def build_extreme_c_splits(
    label_type: str,
    y_all: np.ndarray,
    meta: pd.DataFrame,
    pooled_splits: list,
    train_rows: set,
    top_ct_quantile: float = 0.33,
) -> list:
    """Build extreme-C splits for any binary label.

    Extreme pool E: label=1 patients from cancer types in the top quantile
    by label prevalence (same design principle as prepare_data.py).
    A'/B reuse the matched cohort_aprime/cohort_b from pooled_splits so the
    reference distribution is always balanced pan-cancer.
    C is sampled from E (same size as the pooled split's cohort_c).

    Returns list of dicts with split_id, cohort_aprime, cohort_b, cohort_c.
    """
    held = ~meta['row_idx'].isin(train_rows)
    ct_arr = meta['cancer_type'].values

    # Cancer-type prevalence among held-out valid patients
    held_valid = np.where(held.values & (y_all >= 0))[0]
    ct_prev: dict = {}
    for ct in np.unique(ct_arr[held_valid]):
        mask = ct_arr[held_valid] == ct
        yct = y_all[held_valid[mask]]
        if len(yct) >= 10:
            ct_prev[ct] = float(yct.mean())

    if not ct_prev:
        ts(f"  {label_type}: no cancer types with >=10 valid patients — skipping extreme-C")
        return []

    threshold = float(np.quantile(list(ct_prev.values()), 1 - top_ct_quantile))
    extreme_cts = {ct for ct, p in ct_prev.items() if p >= threshold}
    e_mask = (y_all == 1) & held.values & np.isin(ct_arr, list(extreme_cts))
    E_idx = np.where(e_mask)[0]
    ts(f"  {label_type} extreme-C: CTs={sorted(extreme_cts)}, |E|={len(E_idx)}")

    splits = []
    for i, sp in enumerate(pooled_splits):
        rng = np.random.default_rng(i)
        n_c = min(len(E_idx), len(sp['cohort_c']))
        if n_c < 20:
            continue
        splits.append({
            'split_id':      i,
            'cohort_aprime': sp['cohort_aprime'],
            'cohort_b':      sp['cohort_b'],
            'cohort_c':      rng.choice(E_idx, size=n_c, replace=False).tolist(),
        })
    return splits


def ts(m: str) -> None:
    """Print a timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_fm_embeddings(fm_name: str, meta: pd.DataFrame) -> np.ndarray:
    """Load FM embeddings aligned to the evaluation cohort patient order.

    Parameters
    ----------
    fm_name : str
        Key into FM_PATHS identifying which foundation model to load.
    meta : pd.DataFrame
        Patient metadata with columns ``patient_id`` and ``row_idx``.
        The returned array is aligned to this DataFrame's row order.

    Returns
    -------
    np.ndarray
        Float32 array of shape (N, D). Rows for patients without embeddings
        in this FM are filled with NaN.
    """
    path = FM_PATHS[fm_name]
    df   = pd.read_csv(path, low_memory=False)
    id_col = 'USUBJID'
    df['patient_key'] = df[id_col].apply(
        lambda x: '-'.join(str(x).replace('TCGA/', '').split('-')[:3]))
    df = df.drop_duplicates('patient_key').set_index('patient_key')
    ecols = [c for c in df.columns if c.startswith('E')]
    D = len(ecols)

    # Build array aligned to meta row order; missing patients → NaN
    Z = np.full((len(meta), D), np.nan, dtype=np.float32)
    for i, pid in enumerate(meta['patient_id']):
        if pid in df.index:
            Z[i] = df.loc[pid, ecols].values.astype(np.float32)

    n_valid = (~np.isnan(Z[:, 0])).sum()
    ts(f"  {fm_name}: {n_valid}/{len(meta)} patients with embeddings, dim={D}")
    return Z


def signal_gate(
    Z_ap: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    """Run Stage II signal gate via label-permutation test on CV-AUROC.

    Parameters
    ----------
    Z_ap : np.ndarray
        Standardized H&E embeddings for cohort A', shape (N_ap, D).
    y : np.ndarray
        Binary outcome labels for A', shape (N_ap,).
    rng : np.random.Generator
        Random number generator for permutation seeds.

    Returns
    -------
    tuple[bool, float]
        ``(signal_present, auroc_observed)`` where ``signal_present`` is
        True when the permutation p-value is below 0.05.
    """
    splits   = make_cv_splits(len(y), y, n_folds=K_CV, rng=rng)
    observed = cv_auroc(Z_ap, y, splits=splits)
    seeds    = rng.integers(1 << 63, size=N_PERM)
    null_vals = np.array([
        cv_auroc(Z_ap, np.random.default_rng(s).permutation(y), splits=splits)
        for s in seeds
    ])
    p = (1 + np.sum(null_vals >= observed)) / (N_PERM + 1)
    return bool(p < 0.05), float(observed)


def _verdict(signal_present, ptf_fires, dq_fires):
    if not signal_present:
        return 'S3'
    if ptf_fires and dq_fires:
        return 'S2'
    if ptf_fires and not dq_fires:
        return 'Transferable'
    return 'Indeterminate'


def _evaluate_unimodal(
    fm_name, split_id, split_type, label_type, eta2,
    # Pre-extracted per-cohort arrays (small, fast to serialize vs full Z_all)
    Z_ap_std, y_ap_f,
    Z_b_std,  y_b_f,
    Z_c_std,  y_c_f,   # y_c_f may contain -1 (missing) or valid labels
    Z_ap_raw, Z_b_raw, Z_c_raw,
    n_ap, n_b, n_c,
    seed,
) -> dict | None:
    """Run Stages II+IV on pre-extracted cohort arrays.

    Also computes AUROC of the A'-trained probe evaluated on C and label
    prevalence in C, so callers can assess whether AUROC on an external
    deployment cohort would give a practitioner the same information as DECAT.
    AUROC on C is None when C has fewer than 2 patients per class.
    """
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    try:
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(seed)
        sig, auroc = signal_gate(Z_ap_std, y_ap_f,
                                 np.random.default_rng(rng.integers(1 << 63)))
        t_ci   = compute_transfer_ci(Z_ap_std, y_ap_f, Z_b_std, y_b_f, Z_c_std,
                                     n_bootstrap=N_BOOTSTRAP,
                                     rng=np.random.default_rng(rng.integers(1 << 63)))
        t_null = compute_transfer_null(Z_ap_std, y_ap_f, Z_b_std, y_b_f, Z_c_std,
                                       n_permutations=N_PERM,
                                       rng=np.random.default_rng(rng.integers(1 << 63)))
        dq     = compute_dquantile_task(Z_ap_raw, Z_b_raw, Z_c_raw, y_ap_f,
                                        n_permutations=N_PERM,
                                        rng=np.random.default_rng(rng.integers(1 << 63)))
        ptf_fires = bool(t_ci['ci_lower'] > t_null['null_upper'])
        dq_fires  = bool(dq['dquantile_task'] >= dq['dquantile_task_null_upper'])

        # AUROC of A'-trained probe on C (practitioner's external-validation metric)
        auroc_on_c    = None
        prevalence_c  = None
        valid_c = y_c_f >= 0
        y_c_valid = y_c_f[valid_c]
        Z_c_valid = Z_c_std[valid_c]
        if len(y_c_valid) > 0:
            prevalence_c = float(y_c_valid.mean())
            n_pos, n_neg = int(y_c_valid.sum()), int((y_c_valid == 0).sum())
            if n_pos >= 2 and n_neg >= 2:
                _, clf = fit_probe(Z_ap_std, y_ap_f, probe_backend='logistic',
                                   C=0.1, solver='liblinear', max_iter=500,
                                   standardize=False)
                auroc_on_c = round(
                    float(roc_auc_score(y_c_valid,
                                        predict_scores(None, clf, Z_c_valid))), 4)

        return {
            'fm': fm_name, 'fm_type': FM_TYPE[fm_name],
            'split_id': split_id, 'split_type': split_type,
            'label': label_type, 'eta2': eta2,
            'n_ap': n_ap, 'n_b': n_b, 'n_c': n_c,
            'auroc': round(auroc, 4), 'signal_present': sig,
            'auroc_on_c': auroc_on_c, 'prevalence_c': prevalence_c,
            'p_transfer': float(t_ci['p_transfer']),
            'transfer_ci_lower': float(t_ci['ci_lower']),
            'transfer_null_upper': float(t_null['null_upper']),
            'dquantile_task': float(dq['dquantile_task']),
            'dquantile_null_upper': float(dq['dquantile_task_null_upper']),
            'ptf_fires': ptf_fires, 'dq_fires': dq_fires,
            'verdict': _verdict(sig, ptf_fires, dq_fires),
        }
    except Exception as e:
        import traceback
        ts(f"  ERROR {fm_name}/{label_type}/{split_id}: {e}\n{traceback.format_exc()}")
        return None


def main() -> None:
    """Entry point: parse --out-dir and run unimodal FM comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--labels', default=None,
                        help='Comma-separated labels to run, e.g. kras,msi,os. '
                             'Default: tmb,tp53,age')
    parser.add_argument('--suffix', default='',
                        help='Suffix for output filenames, e.g. "_v2" writes '
                             'unimodal_fm_comparison_v2_records.json')
    args = parser.parse_args()

    out_dir      = Path(args.out_dir) if args.out_dir else OUT_DIR
    suffix       = args.suffix
    label_subset = tuple(args.labels.split(',')) if args.labels else DEFAULT_LABELS

    ts(f"=== Unimodal FM comparison  OUT_DIR={out_dir} ===")
    ts(f"Labels: {label_subset}  Suffix: '{suffix}'  N_JOBS={N_JOBS}")

    # Load metadata and splits from out_dir
    meta          = pd.read_csv(out_dir / 'patient_meta.csv')
    task_data     = np.load(out_dir / 'task_labels.npz')
    pooled_splits = json.load(open(out_dir / 'pooled_random_splits.json'))
    train_rows    = set(pooled_splits[0]['cohort_a'])
    ct_arr        = meta['cancer_type'].values

    # Build label dict: task_labels.npz for tmb/tp53/age, external for others
    all_labels: dict = {
        'tmb':  task_data['s2b_tmb'],
        'tp53': task_data['s2b_tp53'],
        'age':  task_data['s2b_age'],
    }
    if 'kras' in label_subset:
        ts("Loading external labels (KRAS)...")
        all_labels.update(load_external_labels(meta))

    # Compute eta2 for each requested label
    label_eta2: dict = dict(LABEL_ETA2)  # seed with known values
    for lt in label_subset:
        if lt not in label_eta2:
            label_eta2[lt] = round(compute_eta2(all_labels[lt], ct_arr), 3)
            ts(f"  η²({lt}) = {label_eta2[lt]:.3f}")

    # Build extreme-C split lists for each requested label
    ec_splits: dict = {}
    for lt in label_subset:
        json_path = out_dir / f'{lt}_extreme_c_splits.json'
        if json_path.exists():
            ec_splits[lt] = json.load(open(json_path))
        else:
            ec_splits[lt] = build_extreme_c_splits(
                lt, all_labels[lt], meta, pooled_splits, train_rows)

    stem        = f'unimodal_fm_comparison{suffix}'
    out_records = out_dir / f'{stem}_records.json'
    out_summary = out_dir / f'{stem}_summary.csv'

    # Load existing records if resuming
    all_records = []
    done_keys   = set()
    if out_records.exists():
        all_records = json.load(open(out_records))
        done_keys   = {(r['fm'], r['split_type'], r['split_id'], r['label'])
                       for r in all_records}
        ts(f"Resuming: {len(all_records)} records already done")

    # Evaluate each FM sequentially; save records after each one
    for fm_name in FM_PATHS:
        ts("")
        ts(f"FM: {fm_name}  ({FM_TYPE[fm_name]})")
        Z_all = load_fm_embeddings(fm_name, meta)

        for label_type in label_subset:
            y_all = all_labels[label_type].astype(np.int8)  # -1=missing

            def _prep_split(sp_list, split_type_prefix, sid_key):
                """Pre-extract and standardize cohort arrays before parallelizing.
                Avoids serializing the full Z_all (28MB+) to each worker."""
                jobs = []
                for sp in sp_list:
                    sid = sp[sid_key]
                    st  = 'pooled' if split_type_prefix == 'pooled' else f'{label_type}_extreme_c'
                    key = (fm_name, st, sid, label_type)
                    if key in done_keys:
                        continue
                    idx_ap = np.array(sp['cohort_aprime'])
                    idx_b  = np.array(sp['cohort_b'])
                    idx_c  = np.array(sp['cohort_c'])

                    # Filter valid patients (have embeddings + valid labels)
                    m_ap = (~np.isnan(Z_all[idx_ap, 0])) & (y_all[idx_ap] >= 0)
                    m_b  = (~np.isnan(Z_all[idx_b,  0])) & (y_all[idx_b]  >= 0)
                    m_c  = (~np.isnan(Z_all[idx_c,  0]))
                    ap, b, c = idx_ap[m_ap], idx_b[m_b], idx_c[m_c]
                    y_ap_f, y_b_f = y_all[ap], y_all[b]
                    # y_c_f includes all C patients with embeddings; labels may be
                    # missing (-1) for some — DECAT metrics don't need C labels,
                    # but we pass them to compute AUROC-on-C where available.
                    y_c_f = y_all[c]

                    if len(ap) < 30 or len(b) < 20 or len(c) < 20:
                        continue
                    if y_ap_f.sum() < K_CV or (len(y_ap_f) - y_ap_f.sum()) < K_CV:
                        continue

                    # Standardize in main process — only small sub-arrays go to workers
                    scaler = StandardScaler().fit(Z_all[ap])
                    seed = int(abs(hash(f"{fm_name}_{st}_{sid}_{label_type}")) % (1 << 63))
                    jobs.append((
                        fm_name, sid, st, label_type, label_eta2.get(label_type),
                        scaler.transform(Z_all[ap]),  y_ap_f.copy(),
                        scaler.transform(Z_all[b]),   y_b_f.copy(),
                        scaler.transform(Z_all[c]),   y_c_f.copy(),
                        Z_all[ap].copy(), Z_all[b].copy(), Z_all[c].copy(),
                        len(ap), len(b), len(c), seed,
                    ))
                return jobs

            # Pooled splits
            jobs = _prep_split(pooled_splits, 'pooled', 'seed')
            if jobs:
                ts(f"  {label_type} pooled: {len(jobs)} splits")
                results = Parallel(n_jobs=N_JOBS, backend='loky')(
                    delayed(_evaluate_unimodal)(*j) for j in jobs)
                new = [r for r in results if r is not None]
                all_records.extend(new)
                ts(f"    → {len(new)} records")

            # Extreme-C splits
            jobs = _prep_split(ec_splits[label_type], label_type, 'split_id')
            if jobs:
                ts(f"  {label_type} extreme-C: {len(jobs)} splits")
                results = Parallel(n_jobs=N_JOBS, backend='loky')(
                    delayed(_evaluate_unimodal)(*j) for j in jobs)
                new = [r for r in results if r is not None]
                all_records.extend(new)
                ts(f"    → {len(new)} records")

        # Save after each FM
        with open(out_records, 'w') as f:
            json.dump(all_records, f)
        ts(f"  Saved {len(all_records)} records so far")

    # Build and save summary table
    ts("Summary:")
    df = pd.DataFrame(all_records)
    rows = []
    for fm in FM_PATHS:
        for lt in label_subset:
            for st, st_label in [('pooled', 'Pooled'), (f'{lt}_extreme_c', 'Extreme-C')]:
                sub = df[(df['fm']==fm) & (df['label']==lt) & (df['split_type']==st)]
                if len(sub) == 0:
                    continue
                auroc_c_vals = sub['auroc_on_c'].dropna() if 'auroc_on_c' in sub else []
                prev_c_vals  = sub['prevalence_c'].dropna() if 'prevalence_c' in sub else []
                rows.append({
                    'FM':              fm,
                    'FM_type':         FM_TYPE[fm],
                    'Label':           lt.upper(),
                    'η²':              label_eta2.get(lt),
                    'Split':           st_label,
                    'N_splits':        len(sub),
                    'AUROC_Ap':        round(sub['auroc'].mean(), 3),
                    'AUROC_C':         round(float(auroc_c_vals.mean()), 3) if len(auroc_c_vals) > 0 else None,
                    'Prevalence_C':    round(float(prev_c_vals.mean()), 3) if len(prev_c_vals) > 0 else None,
                    'Signal%':         round(sub['signal_present'].mean()*100, 1),
                    'Ptf_fires%':      round(sub['ptf_fires'].mean()*100, 1),
                    'Dq_fires%':       round(sub['dq_fires'].mean()*100, 1),
                    'S2%':             round((sub['verdict']=='S2').mean()*100, 1),
                    'Transferable%':   round((sub['verdict']=='Transferable').mean()*100, 1),
                    'S3%':             round((sub['verdict']=='S3').mean()*100, 1),
                    'Indeterminate%':  round((sub['verdict']=='Indeterminate').mean()*100, 1),
                    'Modal_verdict':   sub['verdict'].value_counts().index[0],
                })

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(out_summary, index=False)
    print(df_sum.to_string(index=False))
    ts(f"\nSaved summary → {out_summary}")
    ts("Done.")


if __name__ == '__main__':
    main()
