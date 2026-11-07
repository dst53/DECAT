"""Within-cancer-type AUROC collapse stratified by putative confounder.

Methodology
-----------
For each (cancer_type, label, stratum) experiment, computes:
  - Pan-CT AUROC: probe trained on cohort A' (cohort_a patients of that cancer
    type), evaluated on held-out B. One number per split, averaged over 50 splits.
  - Per-stratum AUROC: same probe evaluated separately within each confounder
    stratum (e.g. CDH1-mutant vs CDH1-wildtype). Reported individually — no
    aggregation across strata.
  - Collapse Δ = pan_auroc − stratum_auroc for each stratum.
  - Prevalence of the positive class within each stratum (for transparency).

This is the **Dawood et al. (2026) methodology** (Nature BME, Fig 1):
report per-stratum AUROC directly, showing which subgroup is affected.
Example: ER prediction in CDH1-mutant patients (lobular, 96% ER+) gives
AUROC≈0.71 vs pan-BRCA AUROC≈0.84 — the collapse is visible in that subgroup
but is masked by any aggregation method.

Contrast with the pan-cancer collapse in Fig 4c (run_driver_gene_collapse.py),
which uses a pair-weighted C-statistic: AUROC per cancer type weighted by
n_pos × n_neg. This script reports per-stratum AUROC with no aggregation, for
comparison with Dawood et al., where subgroup-specific numbers are reported
directly. The within_ct_confounding notebook reports the worst-performing
stratum (maximum Δ).

Putative confounder strata (BRCA):
  TP53, PR → ER status (Positive / Negative)
  CDH1     → Primary diagnosis (Lobular / Ductal)
  ER       → CDH1 mutation status (CDH1+ / CDH1-)

Output:
  <OUT_DIR>/within_ct_collapse_records.json — one record per (cancer, label, fm, split, stratum)
  <OUT_DIR>/within_ct_collapse_summary.csv  — averaged over splits, one row per
                                              (fm, cancer, label, stratum_key, stratum);
                                              stratum='pan' rows hold the pan-CT AUROC

Usage:
  python experiments/tcga_validation/run_within_ct_collapse.py
  python experiments/tcga_validation/run_within_ct_collapse.py --out-dir <dir>
"""

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[2]))
from experiments.tcga_validation.run_unimodal_fm_comparison import load_fm_embeddings

OUT_DIR      = Path(os.environ.get('DECAT_OUT_DIR', './data/tcga'))
CLINICAL_CSV = os.environ.get('DECAT_CLINICAL_CSV', './data/tcga/CLINICAL.csv')
MUTATION_CSV = os.environ.get('DECAT_MUTATION_CSV', './data/tcga/tcga_driver_mutations.csv')
N_SPLITS     = 50
MIN_CT_SIZE  = 5   # min patients per confounder stratum with both classes
FMS_ORD      = ['titan', 'conch', 'uni', 'hoptimus0', 'openmidnight']


def ts(msg: str) -> None:
    """Print a timestamped status message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_labels(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return dict of binary label arrays (-1=missing) for all experiments."""
    pids = meta['patient_id'].values
    pid12 = np.array([p[:12] for p in pids])
    labels = {}

    # ER / PR status
    er_pr = pd.read_csv('./data/tcga/tcga_brca_er_pr_status.csv')
    er_pr['pid12'] = er_pr['patient_id'].str[:12]
    for col, key in [('er_status_ihc','ER'), ('pr_status_ihc','PR')]:
        smap = dict(zip(er_pr['pid12'], er_pr[col]))
        y = np.full(len(meta), -1.0)
        for i, p in enumerate(pid12):
            s = smap.get(p)
            if s == 'Positive': y[i] = 1.0
            elif s == 'Negative': y[i] = 0.0
        labels[key] = y

    # MC3 gene mutations
    muts = pd.read_csv(MUTATION_CSV)
    muts['pid12'] = muts['patient_id'].str[:12]
    for gene in ['TP53', 'CDH1']:
        if gene in muts.columns:
            gmap = dict(zip(muts['pid12'], muts[gene]))
            y = np.full(len(meta), -1.0)
            for i, p in enumerate(pid12):
                v = gmap.get(p, -1)
                if v >= 0: y[i] = float(v)
            labels[gene] = y

    return labels


def build_strata(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return dict of stratum arrays (str per patient, '' = unknown)."""
    pids = meta['patient_id'].values
    pid12 = np.array([p[:12] for p in pids])
    n = len(meta)
    strata = {}

    # ER status stratum
    er_pr = pd.read_csv('./data/tcga/tcga_brca_er_pr_status.csv')
    er_pr['pid12'] = er_pr['patient_id'].str[:12]
    er_map = dict(zip(er_pr['pid12'], er_pr['er_status_ihc']))
    strata['ER_STATUS'] = np.array([
        er_map.get(p,'') if er_map.get(p,'') in ('Positive','Negative') else ''
        for p in pid12])

    # Primary diagnosis stratum (ductal vs lobular)
    # Source: CLINICAL.csv column CLIN_PRIMARY_DIAGNOSIS
    # e.g. "Infiltrating Duct Carcinoma" → Ductal, "Lobular Carcinoma" → Lobular
    clin = pd.read_csv(CLINICAL_CSV, low_memory=False)
    clin['pid12'] = clin['patient_id'].str[:12]
    diag_map = dict(zip(clin['pid12'], clin['CLIN_PRIMARY_DIAGNOSIS']))
    def _diag(p):
        d = str(diag_map.get(p, ''))
        if 'Lobular' in d: return 'Lobular'
        if 'duct' in d.lower(): return 'Ductal'
        return ''
    strata['PRIMARY_DIAG'] = np.array([_diag(p) for p in pid12])

    # CDH1 mutation stratum (lobular vs non-lobular by mutation)
    muts = pd.read_csv(MUTATION_CSV)
    muts['pid12'] = muts['patient_id'].str[:12]
    cdh1_map = dict(zip(muts['pid12'], muts['CDH1']))
    strata['CDH1_MUT'] = np.array([
        'CDH1+' if cdh1_map.get(p,-1)==1 else ('CDH1-' if cdh1_map.get(p,-1)==0 else '')
        for p in pid12])

    return strata


EXPERIMENTS = [
    # (cancer_type, label, stratum_key, confounder_desc, eta2)
    ('BRCA', 'TP53',    'ER_STATUS',    'ER status',          0.320),
    ('BRCA', 'CDH1',    'PRIMARY_DIAG', 'Lobular histology',  0.532),
    ('BRCA', 'ER',      'CDH1_MUT',     'CDH1 mutation',      0.043),
    ('BRCA', 'PR',      'ER_STATUS',    'ER status',          0.482),
]


def compute_collapse(y_all: np.ndarray, stratum_arr: np.ndarray,
                     Z: np.ndarray, ct_idx: np.ndarray,
                     splits: list, min_per_class: int = 5) -> list[dict] | None:
    """Compute pan-CT AUROC and per-stratum AUROC, averaged over splits.

    Matches Dawood et al. methodology: per-stratum AUROC reported directly
    (not aggregated), with prevalence for transparency. Returns one dict per
    stratum, plus a 'pan' entry.

    Parameters
    ----------
    y_all : (N,) float
        Binary labels for all patients (-1 = missing).
    stratum_arr : (N,) str
        Stratum assignment per patient ('' = unknown).
    Z : (N, D) float32
        FM embeddings.
    ct_idx : (N_ct,) int
        Indices of patients in the target cancer type.
    splits : list of dict
        Pooled random splits with 'cohort_aprime' and 'cohort_b' keys.
    min_per_class : int
        Minimum patients per class within a stratum to include it.

    Returns
    -------
    list of dict or None
        One record per stratum (including a 'pan' summary record), or None
        if insufficient data across all splits.
    """
    pan_aurocs: list[float] = []
    # per-stratum: stratum_label → list of (auroc, prevalence, n)
    stratum_data: dict[str, list[tuple[float, float, int]]] = {}

    for sp in splits:
        ap_r = np.array(sp['cohort_aprime'])
        b_r  = np.array(sp['cohort_b'])

        ap_ct = ap_r[np.isin(ap_r, ct_idx)]
        b_ct  = b_r[np.isin(b_r,  ct_idx)]

        v_ap     = (y_all[ap_ct] >= 0) & ~np.isnan(Z[ap_ct, 0])
        v_b      = (y_all[b_ct]  >= 0) & ~np.isnan(Z[b_ct,  0])
        ap_idx   = ap_ct[v_ap]
        b_idx    = b_ct[v_b]
        ev       = ct_idx[~np.isin(ct_idx, ap_r)]
        eval_idx = ev[(y_all[ev] >= 0) & ~np.isnan(Z[ev, 0])]

        if len(ap_idx) < 20 or len(b_idx) < 10 or len(eval_idx) < 20:
            continue
        y_ap = y_all[ap_idx]
        if (y_ap == 1).sum() < 3 or (y_ap == 0).sum() < 3:
            continue

        scaler = StandardScaler().fit(Z[ap_idx])
        clf = LogisticRegression(C=0.1, max_iter=500, solver='liblinear')
        clf.fit(scaler.transform(Z[ap_idx]).astype(np.float32), y_ap)

        y_b = y_all[b_idx]
        if len(np.unique(y_b)) < 2:
            continue
        pan_aurocs.append(float(roc_auc_score(
            y_b, clf.decision_function(scaler.transform(Z[b_idx]).astype(np.float32)))))

        # Per-stratum AUROC (Dawood methodology)
        y_ev = y_all[eval_idx]
        s_ev = clf.decision_function(scaler.transform(Z[eval_idx]).astype(np.float32))
        strata_ev = stratum_arr[eval_idx]
        for s in np.unique(strata_ev):
            if s == '':
                continue
            m = strata_ev == s
            yt, st = y_ev[m], s_ev[m]
            np_, nn = int((yt == 1).sum()), int((yt == 0).sum())
            if np_ < min_per_class or nn < min_per_class:
                continue
            auc  = float(roc_auc_score(yt, st))
            prev = float(np_ / (np_ + nn))
            stratum_data.setdefault(s, []).append((auc, prev, np_ + nn))

    if not pan_aurocs:
        return None

    pm = float(np.mean(pan_aurocs))
    pm_se = float(np.std(pan_aurocs) / np.sqrt(len(pan_aurocs)))

    records = [{'stratum': 'pan',
                'auroc': round(pm, 4), 'auroc_se': round(pm_se, 4),
                'prevalence': None, 'n': None,
                'delta': 0.0, 'n_splits': len(pan_aurocs)}]

    for s, vals in sorted(stratum_data.items()):
        if len(vals) < 5:
            continue
        aurocs = [v[0] for v in vals]
        prevs  = [v[1] for v in vals]
        ns     = [v[2] for v in vals]
        wm     = float(np.mean(aurocs))
        records.append({
            'stratum':    s,
            'auroc':      round(wm, 4),
            'auroc_se':   round(float(np.std(aurocs) / np.sqrt(len(aurocs))), 4),
            'prevalence': round(float(np.mean(prevs)), 3),
            'n':          round(float(np.mean(ns)), 1),
            'delta':      round(pm - wm, 4),
            'n_splits':   len(aurocs),
        })

    return records if len(records) > 1 else None


def main(out_dir: Path, n_splits: int) -> None:
    """Run within-CT AUROC collapse for all FMs and experiments, write JSON + CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / 'within_ct_collapse_records.json'
    out_csv  = out_dir / 'within_ct_collapse_summary.csv'

    ts("=== Within-CT AUROC collapse (confounder-stratified) ===")
    ts(f"  OUT_DIR  = {out_dir}")
    ts(f"  N_SPLITS = {n_splits}")
    ts(f"  FMs      = {FMS_ORD}")

    meta   = pd.read_csv(out_dir / 'patient_meta.csv')
    splits = json.load(open(out_dir / 'pooled_random_splits.json'))[:n_splits]

    ts("\nLoading labels and confounder strata...")
    labels = load_labels(meta)
    strata = build_strata(meta)
    ts(f"  Labels: {sorted(labels.keys())}")
    ts(f"  Strata: {sorted(strata.keys())}")

    records = []
    for fm in FMS_ORD:
        ts(f"\nLoading {fm}...")
        Z = load_fm_embeddings(fm, meta)
        ts(f"  {(~np.isnan(Z[:,0])).sum()}/{len(meta)} patients, dim={Z.shape[1]}")

        for ct, label, stratum_key, confounder, eta2 in EXPERIMENTS:
            if label not in labels:
                ts(f"  {ct}/{label}: label not loaded — skipping")
                continue
            if stratum_key not in strata:
                ts(f"  {ct}/{label}/{stratum_key}: stratum not loaded — skipping")
                continue

            ct_idx = np.where((meta['cancer_type'] == f'TCGA-{ct}').values)[0]
            y_all = labels[label]
            stratum_arr = strata[stratum_key]

            res_list = compute_collapse(y_all, stratum_arr, Z, ct_idx, splits)
            if res_list:
                pan_rec = next(r for r in res_list if r['stratum'] == 'pan')
                for rec in res_list:
                    records.append({
                        'fm': fm, 'cancer': ct, 'label': label,
                        'stratum_key': stratum_key, 'confounder': confounder,
                        'eta2': eta2, **rec,
                    })
                stratum_recs = [r for r in res_list if r['stratum'] != 'pan']
                ts(f"  {ct}/{label} [{confounder}]: "
                   f"pan={pan_rec['auroc']:.3f}  " +
                   '  '.join(f"{r['stratum']}={r['auroc']:.3f}"
                             f"(Δ{r['delta']:+.3f},prev={r['prevalence']:.0%})"
                             for r in stratum_recs))
            else:
                ts(f"  {ct}/{label} [{confounder}]: insufficient data")

        # Save after each FM
        with open(out_json, 'w') as f:
            json.dump(records, f, indent=2)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    ts(f"\nSaved {len(records)} records → {out_json}")
    ts(f"Saved summary → {out_csv}")

    # Pivot summary — worst-stratum Δ per (cancer, label, fm)
    non_pan = df[df['stratum'] != 'pan']
    worst = (non_pan.groupby(['cancer','label','confounder','eta2','fm'])['delta']
                    .max().reset_index())
    pivot = worst.pivot_table(
        index=['cancer','label','confounder','eta2'], columns='fm', values='delta')
    ts("\nAUROC collapse Δ (worst stratum) by FM:")
    print(pivot.round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--n-splits', type=int, default=N_SPLITS)
    args = parser.parse_args()
    main(args.out_dir, args.n_splits)
