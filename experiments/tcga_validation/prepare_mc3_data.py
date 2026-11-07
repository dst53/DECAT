"""Prepare TCGA driver mutation matrix and TMB from the MC3 public MAF.

Generates two files used by the TCGA validation pipeline:
  - tcga_driver_mutations.csv  (patient × gene binary matrix, 20 genes)
  - tcga_tmb.csv               (patient_id, n_nonsilent_mutations, tmb_per_mb)

Source
------
Pan-cancer somatic mutation calls from the MC3 project (open access):
  https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc

Download command:
  curl -L -o mc3.v0.2.8.PUBLIC.maf.gz \\
    "https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc"

No dbGaP authorization required (open-access MAF, GRCh37, 753 MB).

Usage
-----
  # Download MAF, then run:
  python experiments/tcga_validation/prepare_mc3_data.py \\
      --maf /path/to/mc3.v0.2.8.PUBLIC.maf.gz \\
      --out-dir /path/to/output

  # Or provide an already-uncompressed MAF:
  python experiments/tcga_validation/prepare_mc3_data.py \\
      --maf /path/to/mc3.v0.2.8.PUBLIC.maf \\
      --out-dir /path/to/output
"""

import argparse
import gzip
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# filter to 20 canonical driver genes
DRIVER_GENES = [
    "TP53", "KRAS", "PIK3CA", "BRAF", "PTEN", "APC", "IDH1", "ARID1A",
    "CDH1", "CTNNB1", "RB1", "NF1", "FBXW7", "VHL", "GATA3", "MAP3K1",
    "SMAD4", "SETD2", "NPM1", "EGFR",
]

# Variant classifications to exclude (silent / non-coding)
SILENT_CLASSES = {
    "Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank",
    "IGR", "RNA",
}

# Approximate TCGA exome capture size (Mb) for TMB normalization
EXOME_MB = 38.0


def ts(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def patient_key(barcode: str) -> str:
    """TCGA-XX-XXXX from full sample barcode."""
    return "-".join(str(barcode).split("-")[:3])


def load_maf(maf_path: Path) -> pd.DataFrame:
    """Load MC3 MAF, returning only columns needed for processing."""
    ts(f"Loading MAF from {maf_path} ...")
    opener = gzip.open if str(maf_path).endswith(".gz") else open
    needed = {"Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"}

    chunks = []
    with opener(maf_path, "rt") as fh:
        # Skip comment lines
        line = fh.readline()
        while line.startswith("#"):
            line = fh.readline()
        header = line.rstrip("\n").split("\t")

        col_idx = {c: i for i, c in enumerate(header) if c in needed}
        if not needed.issubset(col_idx):
            missing = needed - set(col_idx)
            raise ValueError(f"MAF missing required columns: {missing}")

        ts(f"  Columns found: {list(col_idx.keys())}")
        for chunk_df in pd.read_csv(
            fh,
            sep="\t",
            header=None,
            names=header,
            usecols=list(needed),
            low_memory=False,
            chunksize=500_000,
        ):
            chunks.append(chunk_df)

    df = pd.concat(chunks, ignore_index=True)
    ts(f"  Loaded {len(df):,} mutation records")
    return df


def make_driver_mutations(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Build patient × gene binary mutation matrix for DRIVER_GENES."""
    ts("Building driver mutation matrix ...")

    # Filter to driver genes and non-silent variants
    dg = df[
        df["Hugo_Symbol"].isin(DRIVER_GENES)
        & ~df["Variant_Classification"].isin(SILENT_CLASSES)
    ].copy()
    dg["patient_id"] = dg["Tumor_Sample_Barcode"].apply(patient_key)

    ts(f"  Non-silent driver-gene variants: {len(dg):,}")

    # Binary matrix: 1 if patient has ≥1 non-silent mutation in gene
    mat = (
        dg.groupby(["patient_id", "Hugo_Symbol"])
        .size()
        .gt(0)
        .astype(int)
        .unstack(fill_value=0)
        .reindex(columns=DRIVER_GENES, fill_value=0)
        .reset_index()
    )
    # Sort genes alphabetically to match canonical column order
    gene_cols = sorted(c for c in mat.columns if c != "patient_id")
    mat = mat[["patient_id"] + gene_cols]

    mat.to_csv(out_path, index=False)
    ts(f"  Saved {len(mat):,} patients × {len(DRIVER_GENES)} genes → {out_path}")

    # Summary
    ts("  Gene mutation rates:")
    for gene in DRIVER_GENES:
        n = int(mat[gene].sum())
        pct = 100 * n / len(mat)
        ts(f"    {gene:<12}  {n:5d}  ({pct:.1f}%)")

    return mat


def make_tmb(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Compute TMB (mutations per Mb) per patient from non-silent mutations."""
    ts("Computing TMB ...")

    ns = df[~df["Variant_Classification"].isin(SILENT_CLASSES)].copy()
    ns["patient_id"] = ns["Tumor_Sample_Barcode"].apply(patient_key)

    counts = (
        ns.groupby("patient_id")
        .size()
        .reset_index(name="n_nonsilent_mutations")
    )
    counts["tmb_per_mb"] = (counts["n_nonsilent_mutations"] / EXOME_MB).round(2)

    counts.to_csv(out_path, index=False)
    ts(f"  Saved {len(counts):,} patients → {out_path}")

    ts(f"  TMB range: {counts['tmb_per_mb'].min():.1f} – "
       f"{counts['tmb_per_mb'].max():.1f} mut/Mb")
    ts(f"  Median TMB: {counts['tmb_per_mb'].median():.1f} mut/Mb")
    ts(f"  Mean TMB:   {counts['tmb_per_mb'].mean():.1f} mut/Mb")
    med = counts["tmb_per_mb"].median()
    counts["_bin"] = (counts["tmb_per_mb"] >= med).astype(int)
    ts(f"  Binarized at median: "
       f"{int(counts['_bin'].sum()):,} above / "
       f"{int((counts['_bin']==0).sum()):,} below")

    return counts.drop(columns="_bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--maf", required=True, type=Path,
        help="Path to mc3.v0.2.8.PUBLIC.maf.gz (or uncompressed .maf)"
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path,
        help="Output directory (will be created if needed)"
    )
    args = parser.parse_args()

    if not args.maf.exists():
        print(f"ERROR: MAF file not found: {args.maf}")
        print()
        print("Download with:")
        print('  curl -L -o mc3.v0.2.8.PUBLIC.maf.gz \\')
        print('    "https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc"')
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_maf(args.maf)

    make_driver_mutations(df, args.out_dir / "tcga_driver_mutations.csv")
    make_tmb(df, args.out_dir / "tcga_tmb.csv")

    ts("\n=== Done ===")
    ts(f"  tcga_driver_mutations.csv → {args.out_dir / 'tcga_driver_mutations.csv'}")
    ts(f"  tcga_tmb.csv              → {args.out_dir / 'tcga_tmb.csv'}")


if __name__ == "__main__":
    main()
