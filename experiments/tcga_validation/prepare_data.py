"""TCGA Validation — Data Preparation.

Pairs any slide-level H&E pathology FM embedding CSV with bulk RNA-seq
FM embedding CSV for TCGA patients, then builds three cohort split structures
and six task label arrays.

Split structures
----------------
  pooled_random_splits.json
      N_SPLITS random 55/15/15/15 splits of all paired patients (equal cohorts).
      Used for S1 / S4 / S3 / S2B (CCA variate) evaluation.

Task labels (task_labels.npz)
------------------------------
  s1_labels    (K_v, N) — CCA variates at VARIATE_INDICES, binarised at median
  s4_labels    (K_v, N) — JIVE RNA-residual PCs at VARIATE_INDICES
  s3_labels    (10, N)  — random binary, seeds 100–109
  s2b_cca      (K_v, N) — CCA variates at VARIATE_INDICES (same as S1)
  s2b_age      (N,)     — age ≥ global median; −1 = missing
  s2b_tp53     (N,)     — TP53 mutation status; −1 = not in MC3
  s2b_tmb      (N,)     — log(TMB+1) ≥ median; −1 = missing from TMB file
  variate_indices       — 1-D array of variate indices used (e.g. [0,5,…,45])

Usage::

    python prepare_data.py                          # TITAN (default)
    python prepare_data.py --he-csv <path> --out-dir <dir>
    python run_pipeline.py --he-csv <path> --out-dir <dir>
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

# H&E embeddings: any slide-level pathology FM CSV with USUBJID column and E* embedding columns
#   CONCH: <DECAT_DATA_DIR>/conch/conch1_5_20x_MEAN.csv
#   TITAN: <DECAT_DATA_DIR>/titan/titan_20x.csv
_FMB_DIR = os.environ.get('DECAT_DATA_DIR', './data/embeddings/pathology')
HE_CSV       = f'{_FMB_DIR}/titan/titan_20x.csv'

# RNA embeddings: bulk RNA-seq BERT CLS embeddings (TCGA pan-cancer)
RNA_CSV      = os.environ.get('DECAT_RNA_CSV', './data/embeddings/rna/bulkRNA_pancancer_value-basedBERT_96M_v1.0-CLS.csv')

CLINICAL_CSV = os.environ.get('DECAT_CLINICAL_CSV', './data/tcga/CLINICAL.csv')
MUTATION_CSV = os.environ.get('DECAT_MUTATION_CSV', './data/tcga/tcga_driver_mutations.csv')
TMB_CSV      = os.environ.get('DECAT_TMB_CSV', './data/tcga/tcga_tmb.csv')
OUT_DIR      = os.environ.get('DECAT_OUT_DIR', './data/tcga')

N_SPLITS     = 50
N_COMPONENTS = 50

FRAC_A  = 0.55
FRAC_AP = 0.15
FRAC_B  = 0.15
# C gets the remainder (~15%) — all four cohorts equal-sized

# Variate/PC indices for S1, S4, S2B_cca (every 5th of 0-49)
VARIATE_INDICES = list(range(50))   # [0, 1, 2, ..., 49] — all 50 variates

# The 10 highest-volume TCGA submitting sites excluded from Cohort A training.
# Note: This allows TSS cross-site evaluation as a future direction to
# study institution-level confounding as a separate TSS cross-site experiment.
TOP_TSS = ["AA", "BH", "DD", "BP", "CV", "DX", "B0", "HT", "A2", "EE"]

def ts(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _patient_key_he(raw_id: str) -> str:
    return "-".join(raw_id.replace("TCGA/", "").split("-")[:3])


def _patient_key_rna(raw_id: str) -> str:
    return "-".join(raw_id.split("-")[:3])


def _tss_from_key(key: str) -> str:
    return key.split("-")[1]


def load_he(path: str) -> pd.DataFrame:
    ts(f"Loading H&E CSV from {path} ...")
    df = pd.read_csv(path)
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "USUBJID"})
    df["patient_key"] = df["USUBJID"].map(_patient_key_he)
    ts(f"  H&E: {len(df)} rows, {df.shape[1]} cols")
    return df


def load_rna(path: str) -> pd.DataFrame:
    ts(f"Loading RNA CSV from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    id_col = df.columns[-1]
    if "USUBJID" not in id_col.upper():
        cands = [c for c in df.columns if "USUBJID" in c.upper() or "SUBJECT" in c.upper()]
        id_col = cands[0] if cands else df.columns[-1]
    df = df.rename(columns={id_col: "USUBJID"})
    df = df[df["USUBJID"].str.startswith("TCGA")].copy()
    df["patient_key"] = df["USUBJID"].map(_patient_key_rna)
    ts(f"  RNA: {len(df)} rows after TCGA filter")
    return df


def join_modalities(
    df_he: pd.DataFrame,
    df_rna: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    he_dedup  = df_he.drop_duplicates("patient_key").copy()
    rna_dedup = df_rna.drop_duplicates("patient_key").copy()
    common = sorted(set(he_dedup["patient_key"]) & set(rna_dedup["patient_key"]))
    ts(f"  H&E unique patients: {len(he_dedup)}")
    ts(f"  RNA unique patients: {len(rna_dedup)}")
    ts(f"  Intersection: {len(common)} paired patients")

    he_filt  = he_dedup.set_index("patient_key").loc[common]
    rna_filt = rna_dedup.set_index("patient_key").loc[common]

    he_emb  = he_filt[[c for c in he_filt.columns  if c.startswith("E")]].values.astype(np.float32)
    rna_emb = rna_filt[[c for c in rna_filt.columns if c.startswith("E")]].values.astype(np.float32)
    ts(f"  H&E embedding shape:  {he_emb.shape}")
    ts(f"  RNA embedding shape:  {rna_emb.shape}")
    return common, he_emb, rna_emb


def build_patient_meta(keys: list[str]) -> pd.DataFrame:
    tss_list = [_tss_from_key(k) for k in keys]
    return pd.DataFrame({
        "patient_id": keys,
        "TSS":        tss_list,
        "row_idx":    list(range(len(keys))),
    })


def build_pooled_random_splits(n: int,
                               row_offset: list[int] | None = None) -> list[dict]:
    """N_SPLITS random 55/15/15/15 splits of N patients.

    row_offset: maps local indices 0..n-1 to actual row indices in the full
    embedding array (pass the list of training-eligible patients).

    Design:
      - Cohort A is fixed once using seed=0.  It is the training set and
        never changes across splits.
      - All N_SPLITS seeds permute ONLY the held-out 45% to produce different
        A'/B/C assignments.  This guarantees that no patient ever appears in
        both Cohort A (training) and any evaluation cohort (A'/B/C) across
        any split.
    """
    pool = np.array(row_offset) if row_offset is not None else np.arange(n)

    # Fix Cohort A once with seed=0
    rng0  = np.random.default_rng(0)
    perm0 = rng0.permutation(n)
    n_a   = int(round(FRAC_A * n))
    cohort_a   = pool[perm0[:n_a]].tolist()
    eval_pool  = pool[perm0[n_a:]].tolist()  # the held-out 45%
    n_eval     = len(eval_pool)
    n_ap       = int(round(FRAC_AP * n))
    n_b        = int(round(FRAC_B  * n))

    splits = []
    for seed in range(N_SPLITS):
        rng   = np.random.default_rng(seed)
        perm  = np.array(eval_pool)[rng.permutation(n_eval)]
        splits.append({
            "seed":          seed,
            "cohort_a":      cohort_a,
            "cohort_aprime": perm[:n_ap].tolist(),
            "cohort_b":      perm[n_ap:n_ap + n_b].tolist(),
            "cohort_c":      perm[n_ap + n_b:].tolist(),
        })
    n_a  = len(splits[0]["cohort_a"])
    n_ap = len(splits[0]["cohort_aprime"])
    n_b  = len(splits[0]["cohort_b"])
    n_c  = len(splits[0]["cohort_c"])
    ts(f"  Pooled random splits: A={n_a}, A'={n_ap}, B={n_b}, C={n_c}")
    ts(f"  Generated {len(splits)} pooled random splits")
    return splits

def build_s2b_cca_variate_labels(Zh_all: np.ndarray) -> np.ndarray:
    """Binarize CCA variates at VARIATE_INDICES global medians.

    Returns shape (len(VARIATE_INDICES), N).
    Same indices as S1/S4 — enables direct comparison of S2B detection
    vs confounding strength (variate index as proxy: k=0 is strongest).
    On pooled splits these labels appear as shared biology (S1) because
    cancer-type distribution is balanced. On extreme-C splits (C = patients
    at extreme CCA variate values, enriched for specific cancer types) the
    same labels reveal confounding (S2).
    """
    labels = np.zeros((len(VARIATE_INDICES), Zh_all.shape[0]), dtype=np.int8)
    for i, k in enumerate(VARIATE_INDICES):
        scores     = Zh_all[:, k]
        global_med = float(np.median(scores))
        labels[i]  = (scores > global_med).astype(np.int8)
        ts(f"  S2B_cca variate {k:2d}: median={global_med:.4f}, "
           f"balance={labels[i].mean():.2f}")
    return labels


def build_s2b_age_label(meta: pd.DataFrame, df_clinical: pd.DataFrame) -> np.ndarray:
    """Age binarized at global median. Returns shape (N,), -1 for missing."""
    n = len(meta)
    vec = np.full(n, -1, dtype=np.int8)
    if df_clinical.empty:
        return vec
    id_col  = next((c for c in df_clinical.columns if "patient" in c.lower() and "id" in c.lower()), None)
    age_col = next((c for c in df_clinical.columns if c.upper() == "CLIN_AGE"), None)
    if not id_col or not age_col:
        return vec
    clin = df_clinical[[id_col, age_col]].copy()
    clin.columns = ["pid", "age"]
    clin["pid"] = clin["pid"].apply(lambda x: "-".join(str(x).split("-")[:3]) if isinstance(x, str) else x)
    clin["age"] = pd.to_numeric(clin["age"], errors="coerce")
    clin = clin.drop_duplicates("pid").set_index("pid")
    ages = meta["patient_id"].map(clin["age"])
    valid = ages.notna()
    med   = ages[valid].median()
    vec[valid.values] = (ages[valid] >= med).astype(np.int8).values
    ts(f"  S2B_age: {valid.sum()}/{n} matched, median={med:.1f}, balance={vec[vec>=0].mean():.2f}")
    return vec


def build_s2b_tp53_label(meta: pd.DataFrame) -> np.ndarray:
    """TP53 mutation status. Returns shape (N,), -1 for patients not in MC3."""
    n   = len(meta)
    vec = np.full(n, -1, dtype=np.int8)
    try:
        muts     = pd.read_csv(MUTATION_CSV).set_index("patient_id")
        common   = sorted(set(meta["patient_id"]) & set(muts.index))
        row_idx  = meta.reset_index().set_index("patient_id").loc[common, "index"].values
        vec[row_idx] = muts.loc[common, "TP53"].values.astype(np.int8)
        n_valid  = (vec >= 0).sum()
        ts(f"  S2B_tp53: {n_valid}/{n} patients with MC3 data, "
           f"prevalence={vec[vec>=0].mean():.3f}")
    except FileNotFoundError:
        ts(f"  S2B_tp53: {MUTATION_CSV} not found — returning -1 for all")
    return vec


def build_s2b_tmb_label(meta: pd.DataFrame) -> np.ndarray:
    """Tumour mutational burden (TMB) binarized at per-cohort median of log(TMB+1).

    Uses log transformation to reduce the influence of hypermutators (UCEC, SKCM).
    Returns shape (N,), -1 for patients not in the TMB file.
    """
    n   = len(meta)
    vec = np.full(n, -1, dtype=np.int8)
    try:
        tmb      = pd.read_csv(TMB_CSV).set_index("patient_id")
        common   = sorted(set(meta["patient_id"]) & set(tmb.index))
        row_idx  = meta.reset_index().set_index("patient_id").loc[common, "index"].values
        log_vals = np.log1p(tmb.loc[common, "tmb_per_mb"].values.astype(np.float32))
        median   = float(np.median(log_vals))
        vec[row_idx] = (log_vals >= median).astype(np.int8)
        n_valid  = (vec >= 0).sum()
        ts(f"  S2B_tmb: {n_valid}/{n} patients with TMB data, "
           f"log-median={median:.3f}, balance={vec[vec>=0].mean():.2f}")
    except FileNotFoundError:
        ts(f"  S2B_tmb: {TMB_CSV} not found — returning -1 for all")
    return vec


def build_task_labels(
    he_emb: np.ndarray,
    rna_emb: np.ndarray,
    cohort_a_idx: list[int],
    meta: pd.DataFrame,
    df_clinical: pd.DataFrame,
) -> dict:
    """Build task labels by fitting CCA/JIVE on Cohort A only, then projecting all patients.

    Parameters
    ----------
    he_emb : ndarray (N, D_h)  — all paired patients
    rna_emb : ndarray (N, D_r) — all paired patients
    cohort_a_idx : list[int]   — row indices for Cohort A (training)
    """
    from models.cca import CCA
    from models.jive import JIVE

    n = he_emb.shape[0]
    a_idx = np.array(cohort_a_idx)
    he_a  = he_emb[a_idx]
    rna_a = rna_emb[a_idx]

    ts(f"  Fitting CCA on Cohort A ({len(a_idx)} patients) for S1 labels ...")
    cca    = CCA(n_components=N_COMPONENTS)
    cca.fit(he_a, rna_a)
    Zh_all = cca.encode_h(he_emb)
    # S1: every 5th variate (VARIATE_INDICES), binarized at median
    s1_labels = np.stack(
        [(Zh_all[:, k] > np.median(Zh_all[:, k])).astype(np.int8)
         for k in VARIATE_INDICES]
    )  # (len(VARIATE_INDICES), N)
    ts(f"  S1 labels: {s1_labels.shape}  indices={VARIATE_INDICES}")

    ts(f"  Fitting JIVE on Cohort A ({len(a_idx)} patients) for S4 labels ...")
    jive        = JIVE(n_components=N_COMPONENTS)
    jive.fit(he_a, rna_a)
    Zr_all      = jive.encode_r(rna_emb)
    z_rna_indiv = Zr_all[:, N_COMPONENTS:]
    s4_labels   = np.stack(
        [(z_rna_indiv[:, k] > np.median(z_rna_indiv[:, k])).astype(np.int8)
         for k in VARIATE_INDICES]
    )  # (len(VARIATE_INDICES), N)
    ts(f"  S4 labels: {s4_labels.shape}")

    s3 = np.zeros((10, n), dtype=np.int8)
    for i, seed in enumerate(range(100, 110)):
        s3[i] = np.random.default_rng(seed).integers(0, 2, size=n, dtype=np.int8)
    ts(f"  S3 labels: {s3.shape}")

    ts("  Building S2B_cca labels (CCA variates at VARIATE_INDICES) ...")
    s2b_cca = build_s2b_cca_variate_labels(Zh_all)

    ts("  Building S2B_age label ...")
    s2b_age = build_s2b_age_label(meta, df_clinical)

    ts("  Building S2B_tp53 label ...")
    s2b_tp53 = build_s2b_tp53_label(meta)

    ts("  Building S2B_tmb label (log TMB binarized at median) ...")
    s2b_tmb = build_s2b_tmb_label(meta)

    return {
        "s1_labels":    s1_labels,    # (K_v, N) where K_v = len(VARIATE_INDICES)
        "s4_labels":    s4_labels,    # (K_v, N)
        "s3_labels":    s3,           # (10, N)
        "s2b_cca":      s2b_cca,      # (K_v, N) — same indices as S1/S4
        "s2b_age":      s2b_age,      # (N,)  -1=missing
        "s2b_tp53":     s2b_tp53,     # (N,)  -1=missing
        "s2b_tmb":      s2b_tmb,      # (N,)  -1=missing; log(TMB+1) ≥ median
        "_cca":         cca,
        "_jive":        jive,
    }


def main() -> None:
    ts("=== TCGA Validation: Step 1 — Data Preparation ===")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_he  = load_he(HE_CSV)
    df_rna = load_rna(RNA_CSV)
    try:
        df_clinical = pd.read_csv(CLINICAL_CSV, low_memory=False)
        ts(f"  Clinical: {len(df_clinical)} rows")
    except FileNotFoundError:
        ts("  Clinical CSV not found; S2B will be zeros")
        df_clinical = pd.DataFrame()

    ts("Joining modalities on patient key ...")
    keys, he_emb, rna_emb = join_modalities(df_he, df_rna)
    n = len(keys)

    ts("Building patient metadata ...")
    meta = build_patient_meta(keys)

    # Attach cancer type from clinical for cross-site splits
    if not df_clinical.empty:
        id_col = next((c for c in df_clinical.columns
                       if "patient" in c.lower() and "id" in c.lower()), None)
        if id_col:
            df_clinical["patient_key"] = df_clinical[id_col].apply(
                lambda x: "-".join(str(x).split("-")[:3]) if isinstance(x, str) else x)
            ci = df_clinical.drop_duplicates("patient_key").set_index("patient_key")
            if "CLIN_PROJECT_ID" in ci.columns:
                meta["cancer_type"] = meta["patient_id"].map(ci["CLIN_PROJECT_ID"])

    ts("Saving embeddings ...")
    np.save(out_dir / "he_embeddings.npy",  he_emb)
    np.save(out_dir / "rna_embeddings.npy", rna_emb)
    meta.to_csv(out_dir / "patient_meta.csv", index=False)
    ts(f"  he_embeddings.npy:  {he_emb.shape}")
    ts(f"  rna_embeddings.npy: {rna_emb.shape}")

    # Exclude TOP_TSS submitting-site patients from the training pool.
    # These were originally reserved to later look at TSS cross-site evaluation.
    eval_site_rows = set(
        meta[meta["TSS"].isin(TOP_TSS)]["row_idx"].tolist()
    )
    non_top_tss_rows = [i for i in range(n) if i not in eval_site_rows]
    ts(f"  TOP_TSS patients (excluded from training): {len(eval_site_rows)} "
       f"({len(eval_site_rows)/n*100:.1f}%)")
    ts(f"  Training-eligible patients:                {len(non_top_tss_rows)}")

    # Cohort A (training) = 55% of training-eligible patients.
    # A'/B/C (evaluation) = held-out 45% of training-eligible patients.
    ts("Building pooled random splits ...")
    pooled_splits = build_pooled_random_splits(len(non_top_tss_rows),
                                               row_offset=non_top_tss_rows)
    with open(out_dir / "pooled_random_splits.json", "w") as f:
        json.dump(pooled_splits, f)
    ts(f"  Saved pooled_random_splits.json ({len(pooled_splits)} splits)")

    cohort_a_idx = pooled_splits[0]["cohort_a"]
    ts(f"  Cohort A (seed=0): {len(cohort_a_idx)} patients")

    ts("Building task labels (CCA/JIVE fit on Cohort A) ...")
    label_dict = build_task_labels(he_emb, rna_emb, cohort_a_idx, meta, df_clinical)
    label_dict.pop("_cca")
    label_dict.pop("_jive")

    np.savez(
        out_dir / "task_labels.npz",
        s1_labels  = label_dict["s1_labels"],   # (K_v, N)
        s4_labels  = label_dict["s4_labels"],   # (K_v, N)
        s3_labels  = label_dict["s3_labels"],   # (10, N)
        s2b_cca    = label_dict["s2b_cca"],     # (K_v, N)  same variate indices as S1
        s2b_age    = label_dict["s2b_age"],     # (N,)      -1=missing
        s2b_tp53   = label_dict["s2b_tp53"],    # (N,)      -1=missing
        s2b_tmb    = label_dict["s2b_tmb"],     # (N,)      -1=missing
        variate_indices = np.array(VARIATE_INDICES),  # saved for reference
    )
    ts("  Saved task_labels.npz")

    ts("=== Step 1 complete ===")
    ts(f"  All paired patients:                      {n}")
    ts(f"  TOP_TSS patients excluded from training:  {len(eval_site_rows)}")
    ts(f"  Training-eligible patients:               {len(non_top_tss_rows)}")
    ts(f"  Cohort A (seed-0):                        {len(cohort_a_idx)} (55% of training-eligible)")
    ts(f"  H&E dim: {he_emb.shape[1]},  RNA dim: {rna_emb.shape[1]}")
    ts(f"  Variate indices: {VARIATE_INDICES}")
    ts(f"  S1/S4/S2B_cca shape: {label_dict['s1_labels'].shape}")
    ts(f"  S2B_age:  {int((label_dict['s2b_age']>=0).sum())} patients with data")
    ts(f"  S2B_tp53: {int((label_dict['s2b_tp53']>=0).sum())} patients with MC3 data")
    ts(f"  S2B_tmb:  {int((label_dict['s2b_tmb']>=0).sum())} patients with TMB data")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--he-csv",  default=None)
    parser.add_argument("--rna-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    a = parser.parse_args()
    if a.he_csv:  HE_CSV  = a.he_csv
    if a.rna_csv: RNA_CSV = a.rna_csv
    if a.out_dir: OUT_DIR = a.out_dir
    main()