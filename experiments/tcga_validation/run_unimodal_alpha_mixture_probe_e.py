"""Driver gene alpha-mixture sweep using probe-score-defined E (all MC3 genes).

Each FM defines its own E from its own probe — no cancer-type labels needed.
This is the correct prospective design: a FM that learned the cancer-type confound
will concentrate E in confounded cancer types (high S2); one that learned genuine
within-type biology will have a more spread E (lower S2). The comparison between
FMs is then meaningful: each FM is evaluated on its own terms.

Gene selection: same criteria as run_driver_gene_collapse.py (~16 genes).

This generates the data for:
  - fig_tcga_driver_gene_ptf_auc_vs_delta  (P_transfer AUC vs Δ, all genes × FMs)
  - fig_tcga_driver_gene_balanced_s2_vs_delta  (S2% vs Δ, driver gene landscape)
  - fig_tcga_unimodal_alpha_mixture  (TP53/KRAS alpha sweep curves)

using a fully prospective E definition — no confounder labels required at any stage.

Output:
  <OUT_DIR>/driver_gene_alpha_mixture_probe_e_records.json

Usage:
  python experiments/tcga_validation/run_unimodal_alpha_mixture_probe_e.py
  python experiments/tcga_validation/run_unimodal_alpha_mixture_probe_e.py \\
      --out-dir ./data/tcga
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[2]))
from evaluation.probes import fit_probe, predict_scores

from experiments.tcga_validation.run_alpha_mixture_sweep_probe_e import (
    _load_globals, make_mixture_splits, N_SEEDS,
)
from experiments.tcga_validation.run_unimodal_fm_comparison import (
    _evaluate_unimodal, load_fm_embeddings, FM_PATHS,
)
from experiments.tcga_validation.run_driver_gene_collapse import select_genes

_DEFAULT_OUT_DIR = Path(os.environ.get('DECAT_OUT_DIR', './data/tcga'))
MUTATION_CSV     = os.environ.get('DECAT_MUTATION_CSV', './data/tcga/tcga_driver_mutations.csv')
ALPHAS_SUBSET    = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PROBE_DECILE     = 0.90   # top-10%
MIN_CT_MINORITY  = 40
MIN_PREV         = 0.03
N_JOBS           = max(1, os.cpu_count() - 2)


def ts(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_probe_e(gene: str, ho: pd.DataFrame, Z: np.ndarray,
                  splits: list, meta: pd.DataFrame) -> tuple:
    """Define E = top-PROBE_DECILE scoring patients from A' probe.

    Returns (mask_probe_s, top_cts) where mask_probe_s is a
    pd.Series (bool, indexed like ho) compatible with make_mixture_splits.
    """
    ap_idx  = np.array(splits[0]['cohort_aprime'])
    ho_ridx = ho['row_idx'].values
    ap_ho   = np.where(np.isin(ho_ridx, ap_idx))[0]
    y_ap    = ho['label'].values[ap_ho].astype(np.int8)
    Z_ap    = Z[ho_ridx[ap_ho]]
    valid   = ~np.isnan(Z_ap[:, 0]) & (y_ap >= 0)
    ap_v    = ap_ho[valid]; y_v = y_ap[valid]; Z_v = Z_ap[valid]

    sc, probe = fit_probe(Z_v, y_v, probe_backend='logistic',
                          C=0.1, solver='liblinear', max_iter=500, random_state=42)

    Z_ho   = Z[ho_ridx]
    valid2 = ~np.isnan(Z_ho[:, 0])
    scores = np.full(len(ho), -np.inf)
    scores[valid2] = predict_scores(sc, probe, Z_ho[valid2])

    thresh      = np.percentile(scores[scores > -np.inf], PROBE_DECILE * 100)
    mask_probe  = scores >= thresh

    # Cancer-type composition for logging
    top_cts = ho[mask_probe].groupby('cancer_type').size().sort_values(ascending=False).head(5)


    ts(f"  {gene} probe-score E: |E|={mask_probe.sum()}, top CTs: {dict(top_cts)}")

    return pd.Series(mask_probe, index=ho.index), dict(top_cts)


def run_one_fm(label_type: str, ho: pd.DataFrame, mask_probe_s: pd.Series,
               fm_name: str, Z_all: np.ndarray) -> list:
    """Alpha-mixture sweep for one label × one FM using that FM's probe-score E."""
    all_splits = make_mixture_splits(ho, mask_probe_s, label_type)
    splits     = [sp for sp in all_splits if sp['alpha'] in set(ALPHAS_SUBSET)]

    jobs, job_as = [], []
    for sp in splits:
        a = sp['alpha']; seed = sp['seed']
        idx_ap = sp['idx_ap']; idx_b = sp['idx_b']; idx_c = sp['idx_c']
        y_ap_f = sp['y_ap'];   y_b_f = sp['y_b'];   y_c_f = sp['y_c']
        m_ap = ~np.isnan(Z_all[idx_ap, 0]); m_b = ~np.isnan(Z_all[idx_b, 0])
        m_c  = ~np.isnan(Z_all[idx_c,  0])
        ap = idx_ap[m_ap]; b = idx_b[m_b]; c = idx_c[m_c]
        y_ap_f = y_ap_f[m_ap]; y_b_f = y_b_f[m_b]; y_c_f = y_c_f[m_c]
        if len(ap) < 30 or len(b) < 20 or len(c) < 20: continue
        if y_ap_f.sum() < 5 or (len(y_ap_f) - y_ap_f.sum()) < 5: continue
        sc2    = StandardScaler().fit(Z_all[ap])
        j_seed = int(abs(hash(f"{fm_name}_{label_type}_{a}_{seed}_probe_e")) % (1 << 63))
        jobs.append((fm_name, seed, f'{label_type}_mix_a{int(a*100):03d}',
                     label_type, None,
                     sc2.transform(Z_all[ap]), y_ap_f,
                     sc2.transform(Z_all[b]),  y_b_f,
                     sc2.transform(Z_all[c]),  y_c_f,
                     Z_all[ap].copy(), Z_all[b].copy(), Z_all[c].copy(),
                     len(ap), len(b), len(c), j_seed))
        job_as.append(a)

    if not jobs:
        return []

    results = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_evaluate_unimodal)(*j) for j in jobs)

    records = []
    for a, rec in zip(job_as, results):
        if rec is not None:
            rec['alpha'] = a
            records.append(rec)
    ts(f"      → {len(records)} records")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--out-dir',         type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument('--min-ct-minority', type=int,  default=MIN_CT_MINORITY)
    parser.add_argument('--genes', nargs='+', default=None, metavar='GENE',
                        help='Subset of genes to run (default: all). '
                             'E.g. --genes MAP3K1 NF1 PIK3CA PTEN RB1 TP53 VHL')
    parser.add_argument('--out-suffix', type=str, default='',
                        help='Suffix for output filename, e.g. "_m2" → '
                             'driver_gene_alpha_mixture_probe_e_m2_records.json')
    global N_JOBS
    parser.add_argument('--n-jobs', type=int, default=N_JOBS)
    args = parser.parse_args()
    N_JOBS  = args.n_jobs
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'driver_gene_alpha_mixture_probe_e{args.out_suffix}_records.json'

    ts("=== Driver gene alpha-mixture sweep (probe-score E, per-FM) ===")
    ts(f"  OUT_DIR  = {out_dir}")
    ts(f"  ALPHAS   = {ALPHAS_SUBSET}")
    ts(f"  E defn   = top-{(1-PROBE_DECILE)*100:.0f}% probe scores per FM (no cancer-type labels)")

    _load_globals(out_dir)
    from experiments.tcga_validation.run_alpha_mixture_sweep_probe_e import ho_all

    meta    = pd.read_csv(out_dir / 'patient_meta.csv')
    muts    = pd.read_csv(MUTATION_CSV).set_index('patient_id')
    splits  = json.load(open(out_dir / 'pooled_random_splits.json'))[:N_SEEDS]

    # Gene selection — same criteria as run_driver_gene_collapse.py
    train_idx = set(splits[0]['cohort_a'])
    held_meta = meta[~meta['row_idx'].isin(train_idx)].copy()
    ts("\nSelecting genes...")
    selected = select_genes(muts, held_meta,
                            min_prev=MIN_PREV,
                            min_ct_minority=args.min_ct_minority)
    if args.genes:
        selected = {g: v for g, v in selected.items() if g in args.genes}
        ts(f"  Filtered to {len(selected)} genes: {sorted(selected.keys())}")
    else:
        ts(f"  {len(selected)} genes: {sorted(selected.keys())}")

    # Load FM embeddings once
    ts("\nLoading FM embeddings...")
    Z_by_fm = {}
    for fm_name in FM_PATHS:
        Z_by_fm[fm_name] = load_fm_embeddings(fm_name, meta)
        ts(f"  {fm_name}: {(~np.isnan(Z_by_fm[fm_name][:,0])).sum()}/{len(meta)}")

    # Resume: load any records already saved and skip completed genes
    done_genes = set()
    if out_json.exists():
        all_records = json.load(open(out_json))
        done_genes  = set(pd.DataFrame(all_records)['label'].unique()) if all_records else set()
        ts(f"Resuming: {len(all_records)} records already saved, "
           f"skipping {len(done_genes)} completed genes: {sorted(done_genes)}")
    else:
        all_records = []

    for gene in sorted(selected.keys()):
        if gene in done_genes:
            ts(f"  Skipping {gene} (already in output)")
            continue
        ts(f"\n{'='*60}  {gene}  (η²={selected[gene]['eta2']:.3f})")
        ho = ho_all.copy()
        ho['label'] = ho['patient_id'].map(muts[gene])
        ho = ho[ho['label'].notna()].reset_index(drop=True)
        ho['label'] = ho['label'].astype(np.int8)
        ts(f"  N={len(ho)}, prev={ho['label'].mean():.1%}")

        gene_records = []
        for fm_name, Z_all in Z_by_fm.items():
            ts(f"  FM: {fm_name}")
            # Each FM defines its own E from its own probe
            mask_probe_s, top_cts = build_probe_e(
                gene, ho, Z_all, splits, meta)
            recs = run_one_fm(gene, ho, mask_probe_s, fm_name, Z_all)
            gene_records.extend(recs)

        all_records.extend(gene_records)
        ts(f"  {gene} total: {len(gene_records)} records")

        # Save incrementally after each gene
        with open(out_json, 'w') as f:
            json.dump(all_records, f)

    ts(f"\nSaved {len(all_records)} records → {out_json}")

    # Quick summary
    df = pd.DataFrame(all_records)
    ts("\nS2% at α=0.4 (strict, probe-score E):")
    for gene in sorted(selected.keys()):
        sub = df[(df['label'] == gene) & (df['alpha'] == 0.4)]
        if len(sub) == 0: continue
        by_fm = {fm: f"{sub[sub['fm']==fm]['verdict'].eq('S2').mean()*100:.0f}%"
                 for fm in ['conch','titan','uni','hoptimus0','openmidnight']
                 if len(sub[sub['fm']==fm]) > 0}
        ts(f"  {gene:10s}: {by_fm}")

    ts("=== Done ===")


if __name__ == '__main__':
    main()
