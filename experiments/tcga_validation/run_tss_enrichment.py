"""Within-cancer-type institution (TSS) enrichment of the probe-score extreme pool.

Tests whether patients with extreme FM probe scores concentrate at specific tissue
source sites (TSS) within cancer types, beyond what label prevalence explains.

Design:
  - Probe: logistic regression trained on cohort_aprime of each pooled random split
  - Held-out set: cohort_b ∪ cohort_c; extreme pool E = top 10% by probe score
  - Per cancer type (≥10 patients in and out of E, ≥3 TSS codes with ≥5 patients):
    Cramér's V of the TSS × E contingency table
  - Statistic: Cramér's V summed over eligible cancer types (independent patients)
  - Null: E permuted within label strata (positive / negative) inside each cancer
    type, preserving the label composition of E; 1,000 permutations
  - One p-value per FM × label × split; summarize by the median over 50 splits

Output:
  <OUT_DIR>/tss_enrichment_records.csv

Usage:
  python experiments/tcga_validation/run_tss_enrichment.py
"""

import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from experiments.tcga_validation.run_within_ct_decat import load_fm_embeddings, FM_PATHS

OUT_DIR  = Path(os.environ.get('DECAT_OUT_DIR', './data/tcga'))
LABELS   = {'tmb': 's2b_tmb', 'tp53': 's2b_tp53', 'age': 's2b_age'}
N_PERM   = 1000
N_SPLITS = 50
E_FRAC   = 0.10
MIN_E    = 10
MIN_TSS  = 5
N_JOBS   = max(1, os.cpu_count() - 2)


def cramers_v(codes: np.ndarray, in_e: np.ndarray) -> float:
    """Cramér's V of a TSS × E (k × 2) contingency table.

    Parameters
    ----------
    codes : np.ndarray
        Integer TSS codes in ``0..k-1``, shape (n,).
    in_e : np.ndarray
        Boolean membership in the extreme pool, shape (n,).

    Returns
    -------
    float
        Cramér's V, which for a k × 2 table is sqrt(chi2 / n).
    """
    k = codes.max() + 1
    tab = np.stack([np.bincount(codes[in_e], minlength=k),
                    np.bincount(codes[~in_e], minlength=k)], 1).astype(float)
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    return float(np.sqrt(((tab - exp) ** 2 / exp).sum() / tab.sum()))


def evaluate_split(fm: str, label: str, split: dict, Z: np.ndarray, y: np.ndarray,
                   tss: np.ndarray, ct: np.ndarray, seed: int) -> dict:
    """Summed within-type Cramér's V and its label-stratified permutation p-value.

    Parameters
    ----------
    fm : str
        Foundation model name.
    label : str
        Label name.
    split : dict
        Pooled random split with ``cohort_aprime``, ``cohort_b``, ``cohort_c``.
    Z : np.ndarray
        FM embeddings, shape (n_patients, d); NaN rows for missing patients.
    y : np.ndarray
        Binary labels, shape (n_patients,); negative values mark missing labels.
    tss : np.ndarray
        TSS code per patient, shape (n_patients,).
    ct : np.ndarray
        Cancer type per patient, shape (n_patients,).
    seed : int
        Seed for the permutation null.

    Returns
    -------
    dict
        Record with the summed V, the number of eligible cancer types, the
        permutation p-value, and the 95th percentile of the null.
    """
    ap = np.array(split['cohort_aprime'])
    held = np.concatenate([split['cohort_b'], split['cohort_c']])
    ap = ap[(y[ap] >= 0) & ~np.isnan(Z[ap, 0])]
    held = held[(y[held] >= 0) & ~np.isnan(Z[held, 0])]
    sc = StandardScaler().fit(Z[ap])
    clf = LogisticRegression(C=0.1, max_iter=500, solver='liblinear').fit(sc.transform(Z[ap]), y[ap])
    s = clf.decision_function(sc.transform(Z[held]))
    in_e = s >= np.quantile(s, 1 - E_FRAC)
    yh, tss_h, ct_h = y[held], tss[held], ct[held]

    rng = np.random.default_rng(seed)
    obs, null, n_ct = 0.0, np.zeros(N_PERM), 0
    for c in np.unique(ct_h):
        m = ct_h == c
        if in_e[m].sum() < MIN_E or (~in_e[m]).sum() < MIN_E:
            continue
        tm, em, ym = tss_h[m], in_e[m], yh[m]
        sites, counts = np.unique(tm, return_counts=True)
        keep_sites = sites[counts >= MIN_TSS]
        if len(keep_sites) < 3:
            continue
        keep = np.isin(tm, keep_sites)
        codes = np.unique(tm[keep], return_inverse=True)[1]
        em, ym = em[keep], ym[keep]
        if em.all() or not em.any():
            continue
        obs += cramers_v(codes, em)
        n_ct += 1
        strata = [np.where(ym == v)[0] for v in (0, 1)]
        for j in range(N_PERM):
            perm = em.copy()
            for idx in strata:
                perm[idx] = rng.permutation(perm[idx])
            null[j] += cramers_v(codes, perm)
    return dict(fm=fm, label=label, split=seed, sum_v=obs, n_ct=n_ct,
                null95=float(np.quantile(null, 0.95)),
                p=float((1 + np.sum(null >= obs)) / (N_PERM + 1)))


def main() -> None:
    meta = pd.read_csv(OUT_DIR / 'patient_meta.csv')
    tss = meta['patient_id'].str[5:7].values
    ct = meta['cancer_type'].values
    task_labels = np.load(OUT_DIR / 'task_labels.npz')
    splits = json.load(open(OUT_DIR / 'pooled_random_splits.json'))[:N_SPLITS]

    records = []
    for fm in FM_PATHS:
        Z = load_fm_embeddings(fm, meta['patient_id'].values)
        for label, key in LABELS.items():
            y = task_labels[key]
            records += Parallel(n_jobs=N_JOBS)(
                delayed(evaluate_split)(fm, label, sp, Z, y, tss, ct, i) for i, sp in enumerate(splits))
        print(f'{fm} done', flush=True)
    out = OUT_DIR / 'tss_enrichment_records.csv'
    pd.DataFrame(records).to_csv(out, index=False)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
