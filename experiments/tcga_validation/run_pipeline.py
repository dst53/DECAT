"""TCGA Validation — full pipeline runner.

Runs prepare_data → train_models → run_decat in sequence for any
pathology FM embedding CSV and output directory.

Usage::

    # TITAN (default)
    python experiments/tcga_validation/run_pipeline.py \\
        --he-csv ./data/embeddings/pathology/titan/titan_20x.csv \\
        --out-dir ./data/tcga

    # Dry-run: prepare data only
    python experiments/tcga_validation/run_pipeline.py --steps prepare

    # Skip already-done steps
    python experiments/tcga_validation/run_pipeline.py --steps train eval
"""

import argparse
import subprocess
import os
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).parent

RNA_V1_CSV = os.environ.get("DECAT_RNA_CSV", "./data/embeddings/rna/bulkRNA_pancancer_value-basedBERT_96M_v1.0-CLS.csv")

DEFAULTS = {
    "he_csv":  os.path.join(os.environ.get("DECAT_DATA_DIR", "./data/embeddings/pathology"), "titan/titan_20x.csv"),
    "rna_csv": RNA_V1_CSV,
    "out_dir": os.environ.get("DECAT_OUT_DIR", "./data/tcga"),
}

STEPS = {
    "prepare": SCRIPTS / "prepare_data.py",
    "train":   SCRIPTS / "train_models.py",
    "eval":    SCRIPTS / "run_decat.py",
}


def ts(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_step(script: Path, he_csv: str, out_dir: str, rna_csv: str | None = None) -> None:
    cmd = [
        sys.executable, str(script),
        "--he-csv",  he_csv,
        "--out-dir", out_dir,
    ]
    if rna_csv and script.name == "prepare_data.py":
        cmd += ["--rna-csv", rna_csv]
    ts(f"Running: {script.name}")
    result = subprocess.run(cmd, check=True)
    ts(f"  {script.name} complete (exit {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="TCGA validation pipeline runner")
    parser.add_argument("--he-csv",  default=DEFAULTS["he_csv"],
                        help="Path to H&E embedding CSV (any pathology FM)")
    parser.add_argument("--rna-csv", default=DEFAULTS["rna_csv"],
                        help="Path to RNA embedding CSV")
    parser.add_argument("--out-dir", default=DEFAULTS["out_dir"],
                        help="Output directory for all pipeline artefacts")
    parser.add_argument("--steps", nargs="+",
                        choices=list(STEPS.keys()), default=list(STEPS.keys()),
                        help="Steps to run (default: all)")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ts(f"=== TCGA validation pipeline ===")
    ts(f"  H&E CSV:  {args.he_csv}")
    ts(f"  RNA CSV:  {args.rna_csv}")
    ts(f"  OUT_DIR:  {args.out_dir}")
    ts(f"  Steps:    {args.steps}")

    for step in args.steps:
        run_step(STEPS[step], args.he_csv, args.out_dir, rna_csv=args.rna_csv)

    ts("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
