"""Within-type AUROC collapse across MC3 driver genes — H&E foundation models.

For each canonical driver gene and each H&E FM, computes:
  - Pan-cancer balanced AUROC  (standard evaluation on held-out B cohort)
  - Median within-cancer-type AUROC  (probe trained pan-cancer, evaluated per CT)
  - Collapse Δ = pan − within
  - η²_cancer  (fraction of label variance explained by cancer type)

Gene selection (fully automatic, no hardcoding):
  All 20 canonical MC3 driver genes from tcga_driver_mutations.csv are evaluated.
  Genes are included in the output if they pass two objective criteria in the
  held-out set:
    1. Pan-cancer prevalence ≥ 3%
       Rationale: ensures sufficient minority-class patients for pan-cancer
       probe training. Excludes NPM1 (<3% in solid-tumour TCGA cohort because
       it is a haematological driver not relevant to WSI analysis).
       Note: the dataset contains no gene above 55% prevalence, so no upper
       bound is applied — roc_auc_score is valid at any class balance.
    2. min(N_pos, N_neg) ≥ MIN_CT_MINORITY within at least one cancer type
       Rationale: statistical power for within-type AUROC estimation. This is
       the binding criterion; criterion 1 only removes NPM1.

  Default MIN_CT_MINORITY=40.

Source:
  MC3 v0.2.8 PUBLIC somatic mutation callset
  Mutations from tcga_driver_mutations.csv (binary, non-silent only, GRCh37).

Output:
  <OUT_DIR>/driver_gene_collapse_records.json  — one record per (gene, FM, split)
  <OUT_DIR>/driver_gene_collapse_summary.csv   — collapsed per (gene, FM)

Usage:
  python experiments/tcga_validation/run_driver_gene_collapse.py
  python experiments/tcga_validation/run_driver_gene_collapse.py \\
      --out-dir ./data/tcga \\
      --min-ct-minority 40 \\
      --n-splits 50
"""

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parents[2]))
from evaluation.probes import fit_probe, predict_scores
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


from experiments.tcga_validation.run_unimodal_fm_comparison import load_fm_embeddings

# Defaults
OUT_DIR       = Path(os.environ.get('DECAT_OUT_DIR', './data/tcga'))
MUTATION_CSV  = os.environ.get('DECAT_MUTATION_CSV', './data/tcga/tcga_driver_mutations.csv')
N_SPLITS      = 50          # number of pooled random splits to use
MIN_CT_MINORITY = 40        # min(N+, N-) in best cancer type (held-out set)
MIN_PREV      = 0.03        # pan-cancer min prevalence (excludes NPM1; see docstring)
MIN_CT_SIZE   = 20          # min patients in a cancer type for within-type AUROC
FMS_ORD       = ['titan', 'conch', 'uni', 'hoptimus0', 'openmidnight']
N_JOBS        = max(1, os.cpu_count() - 2)


def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)



# Gene selection
def select_genes(muts: pd.DataFrame, held_meta: pd.DataFrame,
                 min_prev: float, min_ct_minority: int) -> dict:
    """Return {gene: best_ct_info} for genes passing selection criteria.

    Criteria (applied to held-out set only):
      1. Pan-cancer prevalence >= min_prev (excludes NPM1, too rare in solid tumours;
         no upper bound — roc_auc_score is valid at any class balance)
      2. min(N_pos, N_neg) >= min_ct_minority within at least one cancer type
         (binding criterion for statistical power in within-type AUROC)
    """
    held = held_meta.join(muts, on='patient_id')
    genes = [c for c in muts.columns]
    selected = {}
    skipped = {}

    for gene in sorted(genes):
        y = held[gene].dropna()
        n = len(y)
        if n < 50:
            skipped[gene] = f'N={n} < 50'
            continue
        prev = float(y.mean())
        if prev < min_prev:
            skipped[gene] = f'prevalence={prev:.1%} < {min_prev:.0%} (too rare in solid-tumour cohort)'
            continue

        ct = held.loc[held[gene].notna(), 'cancer_type']

        # η²_cancer
        ct_means = y.groupby(ct).mean()
        grand = float(y.mean())
        ss_b = float(sum(((ct_means[c] - grand) ** 2) * (ct == c).sum()
                         for c in ct_means.index))
        eta2 = ss_b / float(((y - grand) ** 2).sum()) if ((y - grand) ** 2).sum() > 0 else 0.0

        # Find best CT by min(N+, N-)
        best_min, best_ct, best_ct_stats = 0, None, {}
        for cancer, grp in y.groupby(ct):
            n_pos = int(grp.sum())
            n_neg = int(len(grp) - grp.sum())
            m = min(n_pos, n_neg)
            if m > best_min:
                best_min = m
                best_ct = cancer
                best_ct_stats = {'cancer_type': cancer, 'n_pos': n_pos,
                                 'n_neg': n_neg, 'min_class': m}

        if best_min < min_ct_minority:
            skipped[gene] = (f'best CT min_class={best_min} < {min_ct_minority} '
                             f'({best_ct}: {best_ct_stats})')
            continue

        selected[gene] = {
            'n_held': n, 'prevalence': round(prev, 4), 'eta2': round(eta2, 4),
            **best_ct_stats
        }

    ts(f"Gene selection: {len(selected)} pass, {len(skipped)} excluded")
    for g, info in sorted(selected.items()):
        ts(f"  ✓ {g:10s}  η²={info['eta2']:.3f}  prev={info['prevalence']:.1%}  "
           f"best CT: {info['cancer_type']}({info['n_pos']}+/{info['n_neg']}-)")
    ts("  Excluded:")
    for g, reason in sorted(skipped.items()):
        ts(f"    ✗ {g:10s}  {reason}")
    return selected


# Within-type collapse computation
def compute_gene_collapse(gene: str, y_all: np.ndarray, Z: np.ndarray,
                          splits: list, ct_arr: np.ndarray,
                          min_ct_size: int = 20,
                          within_statistic: str = 'median') -> dict:
    """For one (gene, FM) pair: pan-cancer AUROC + within-type AUROC.

    Parameters
    ----------
    within_statistic : 'median' or 'cstat'
        'median'  — simple median across cancer types (original behaviour).
        'cstat'   — pooled stratified C-statistic weighted by n_pos × n_neg
                    (equivalent to stratified Wilcoxon; more robust to
                    extreme-prevalence and small cancer types).
    """
    pan_list, within_list = [], []

    for spl in splits:
        ap_r = np.array(spl['cohort_aprime'])
        b_r  = np.array(spl['cohort_b'])
        v_ap = (y_all[ap_r] >= 0) & ~np.isnan(Z[ap_r, 0])
        v_b  = (y_all[b_r]  >= 0) & ~np.isnan(Z[b_r,  0])
        ap = ap_r[v_ap]; yap = y_all[ap].astype(np.int8)
        b  = b_r[v_b];   yb  = y_all[b].astype(np.int8)
        if yap.sum() < 5 or (len(yap) - yap.sum()) < 5:
            continue

        sc, pr = fit_probe(Z[ap], yap, probe_backend='logistic',
                           C=0.1, solver='liblinear', max_iter=500, random_state=42)

        if len(np.unique(yb)) > 1:
            pan_list.append(float(roc_auc_score(
                yb, predict_scores(sc, pr, Z[b]))))

        # Within-type: held-out patients not in A'
        held_mask = (~np.isin(np.arange(len(y_all)), ap_r)
                     & (y_all >= 0) & ~np.isnan(Z[:, 0]))
        hi = np.where(held_mask)[0]
        ct_aurocs, total_pairs, concordant_w = [], 0.0, 0.0
        for cancer in np.unique(ct_arr[hi]):
            m = hi[ct_arr[hi] == cancer]
            m = m[~np.isin(m, ap_r)]
            if len(m) < min_ct_size:
                continue
            yct = y_all[m].astype(np.int8)
            if yct.sum() < 3 or (len(yct) - yct.sum()) < 3:
                continue
            auc = float(roc_auc_score(yct, predict_scores(sc, pr, Z[m])))
            ct_aurocs.append(auc)
            n_pairs = int(yct.sum()) * (len(yct) - int(yct.sum()))
            concordant_w += auc * n_pairs
            total_pairs  += n_pairs
        if within_statistic == 'cstat':
            if total_pairs > 0:
                within_list.append(concordant_w / total_pairs)
        else:
            if ct_aurocs:
                within_list.append(float(np.median(ct_aurocs)))

    if not pan_list:
        return {}

    pan_m = float(np.mean(pan_list))
    pan_se = float(np.std(pan_list) / np.sqrt(len(pan_list)))
    wit_m = float(np.mean(within_list)) if within_list else float('nan')
    wit_se = (float(np.std(within_list) / np.sqrt(len(within_list)))
              if len(within_list) > 1 else 0.0)
    return {
        'auroc_pan':    round(pan_m,  4),
        'auroc_pan_se': round(pan_se, 4),
        'auroc_within':    round(wit_m,  4) if not np.isnan(wit_m) else None,
        'auroc_within_se': round(wit_se, 4),
        'collapse': round(pan_m - wit_m, 4) if not np.isnan(wit_m) else None,
        'n_pan_splits':    len(pan_list),
        'n_within_splits': len(within_list),
    }


# Main
def main(out_dir: Path, min_ct_minority: int, n_splits: int,
         within_statistic: str = 'median', out_suffix: str = ''):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'driver_gene_collapse_records{out_suffix}.json'
    out_csv  = out_dir / f'driver_gene_collapse_summary{out_suffix}.csv'

    ts("=== Driver gene within-type AUROC collapse ===")
    ts(f"  OUT_DIR           = {out_dir}")
    ts(f"  MIN_CT_MINORITY   = {min_ct_minority}")
    ts(f"  N_SPLITS          = {n_splits}")
    ts(f"  within_statistic  = {within_statistic}")
    ts(f"  out_suffix        = '{out_suffix}'")
    ts(f"  FMs               = {FMS_ORD}")

    meta    = pd.read_csv(out_dir / 'patient_meta.csv')
    splits  = json.load(open(out_dir / 'pooled_random_splits.json'))[:n_splits]
    ct_arr  = meta['cancer_type'].values
    muts    = pd.read_csv(MUTATION_CSV).set_index('patient_id')

    # Held-out patients (not in training cohort A of split 0)
    train_idx  = set(splits[0]['cohort_a'])
    held_meta  = meta[~meta['row_idx'].isin(train_idx)].copy()

    # Gene selection 
    ts("\nSelecting genes...")
    selected = select_genes(muts, held_meta,
                            min_prev=MIN_PREV,
                            min_ct_minority=min_ct_minority)

    # Build label vectors (full cohort, -1 = missing)
    gene_labels = {}
    for gene in selected:
        y = np.full(len(meta), -1, dtype=np.int8)
        common = sorted(set(meta['patient_id']) & set(muts.index))
        ridx = meta.reset_index().set_index('patient_id').loc[common, 'index'].values
        y[ridx] = muts.loc[common, gene].values.astype(np.int8)
        gene_labels[gene] = y

    # Load FM embeddings once, loop over genes
    records = []
    for fm in FMS_ORD:
        ts(f"\nLoading {fm} embeddings...")
        Z = load_fm_embeddings(fm, meta)
        ts(f"  {fm}: {(~np.isnan(Z[:,0])).sum()}/{len(meta)} patients, dim={Z.shape[1]}")

        for gene, y_all in gene_labels.items():
            ts(f"  {gene}...", )
            result = compute_gene_collapse(
                gene, y_all, Z, splits, ct_arr, min_ct_size=MIN_CT_SIZE,
                within_statistic=within_statistic)
            if not result:
                ts(f"    → skipped (insufficient splits)")
                continue
            ts(f"    pan={result['auroc_pan']:.3f}±{result['auroc_pan_se']:.3f}  "
               f"within={result.get('auroc_within','N/A')}  "
               f"Δ={result.get('collapse','N/A')}")
            records.append({
                'gene':        gene,
                'fm':          fm,
                'eta2':        selected[gene]['eta2'],
                'prevalence':  selected[gene]['prevalence'],
                'best_ct':     selected[gene]['cancer_type'],
                'best_ct_min': selected[gene]['min_class'],
                **result,
            })

    # Save 
    with open(out_json, 'w') as f:
        json.dump(records, f, indent=2)
    ts(f"\nSaved {len(records)} records → {out_json}")

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    ts(f"Saved summary → {out_csv}")

    # Print summary table
    ts("\nSummary (sorted by η²):")
    for gene in sorted(selected, key=lambda g: -selected[g]['eta2']):
        sub = df[df['gene'] == gene]
        ts(f"  {gene:10s}  η²={selected[gene]['eta2']:.3f}  "
           + '  '.join(f"{r['fm'][:5]}:pan={r['auroc_pan']:.3f},Δ={r.get('collapse','?')}"
                       for _, r in sub.iterrows()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--min-ct-minority', type=int, default=MIN_CT_MINORITY,
                        help='min(N+,N-) in best cancer type (default: 40)')
    parser.add_argument('--n-splits', type=int, default=N_SPLITS,
                        help='number of pooled random splits to evaluate (default: 50)')
    parser.add_argument('--within-statistic', default='median',
                        choices=['median', 'cstat'],
                        help='within-type estimator: median (default) or pooled C-stat')
    parser.add_argument('--out-suffix', default='',
                        help='suffix for output filenames, e.g. _cstat (default: none)')
    args = parser.parse_args()
    main(args.out_dir, args.min_ct_minority, args.n_splits,
         args.within_statistic, args.out_suffix)
