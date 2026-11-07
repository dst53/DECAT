"""Pre Step A — Alignment saturation study.

Determines the minimal N_train for which A_norm* stabilizes on held-out
Cohort A' across all model classes and latent dimensionalities.

Design:
  - k in {5, 10, 50, 100}
  - N_train in {10k, 20k, 30k, 40k, 50k}  (multiples of latent dim)
  - 6 model configs: CCA, CLIP, JIVE, DSSL(0.01), DSSL(0.1), DSSL(1.0)
  - 5 seeds per (k, N_train)
  - d_obs = 1024, baseline measurement regime
  - Cohort A generated at max(N_train)=50k, subsampled for smaller sizes
  - Cohort A' fixed at 500 patients

Nesting: For each (k, seed), Cohort A is generated at max N_train.
Smaller N_train values use n_a = mult * k with the same seed, so
the first N rows of Cohort A are a deterministic prefix (same rng).
A', B, C are fixed size and identical in distribution across N_train for the
same (k, seed).

For factorized models (JIVE, DSSL): only z_c (shared, cols :K) from
latents.npz is used for A_norm* evaluation.

For each (k, N_train, model, seed), computes:
  - A_norm* = (A_obs - A_null_mean) / (1 - A_null_mean) on Cohort A'
  - Permutation null for A_norm (200 permutations)
  - Bootstrap CI for A_norm* (200 resamples)

Output: JSON array of records saved to
  ./results/pre_step_a.json

Two-phase architecture:
  Phase 1 — Train all runs in batches of BATCH_SIZE=8 using train_runs_gpu_pool.
  Phase 2 — Evaluate A_norm* for all (k, N_train, seed, model) in parallel
             using Parallel(n_jobs=32) from saved latents.

Usage:
  python experiments/run_pre_step_a.py
  python experiments/run_pre_step_a.py --k 5 10 --n-seeds 2  # quick test
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from simulator.simulator import SimulatorConfig
from pipeline.train_models import ModelConfig, train_runs_gpu_pool
from metrics.anorm import compute_anorm, compute_anorm_null

DEFAULT_RESULTS = os.environ.get("DECAT_RESULTS_DIR", "./results")

SEED       = 42
D_OBS      = 1024
N_A_PRIME  = 500
N_PERMS    = 200
N_BOOT     = 200
BATCH_SIZE = 8


# Model configs

def model_configs_for_k(k: int) -> list[ModelConfig]:
    """7 model configs per k: CCA, CLIP, JIVE, all 4 DSSL diagonal configs.

    All 4 DSSL configs are included so N_train saturation is validated
    for the full main experiment model set.
    """
    return [
        ModelConfig("cca",  f"cca_K{k}",              {"n_components": k}, False),
        ModelConfig("clip", f"clip_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4, "temperature": 0.07,
            "lr": 1e-3, "n_epochs": 200, "batch_size": 1024,
        }, False),
        ModelConfig("jive", f"jive_K{k}",
                    {"n_components": k, "n_individual": k}, True),
        ModelConfig("dssl", f"dssl_l0.01_b0.01_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4,
            "beta": 0.01, "lambda_orth": 0.01, "temperature": 0.07,
            "aug_noise_std": 0.1, "lr": 1e-3,
            "n_epochs_shared": 125, "n_epochs_specific": 50,
            "batch_size": 1024,
        }, True),
        ModelConfig("dssl", f"dssl_l0.1_b0.1_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4,
            "beta": 0.1, "lambda_orth": 0.1, "temperature": 0.07,
            "aug_noise_std": 0.1, "lr": 1e-3,
            "n_epochs_shared": 125, "n_epochs_specific": 50,
            "batch_size": 1024,
        }, True),
        ModelConfig("dssl", f"dssl_l1.0_b1.0_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4,
            "beta": 1.0, "lambda_orth": 1.0, "temperature": 0.07,
            "aug_noise_std": 0.1, "lr": 1e-3,
            "n_epochs_shared": 75, "n_epochs_specific": 50,
            "batch_size": 1024,
        }, True),
        ModelConfig("dssl", f"dssl_l10.0_b10.0_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4,
            "beta": 10.0, "lambda_orth": 10.0, "temperature": 0.07,
            "aug_noise_std": 0.1, "lr": 1e-3,
            "n_epochs_shared": 75, "n_epochs_specific": 50,
            "batch_size": 1024,
        }, True),
    ]


# Helpers

def _std(Z: np.ndarray) -> np.ndarray:
    return (Z - Z.mean(0)) / Z.std(0).clip(min=1e-8)


def _run_id(k: int, mult: int, seed_i: int) -> str:
    return f"k{k}_m{mult}_s{seed_i}"


def _shared_rep(lats: np.lib.npyio.NpzFile, key: str,
                k: int, is_factorized: bool) -> np.ndarray:
    """Extract shared representation from latents.npz.

    For factorized models (JIVE, DSSL), latents are [z_c | z_s] of shape
    (n, 2k); only the first k columns are z_c (shared).
    For entangled models (CCA, CLIP), the full array is returned.
    """
    Z = lats[key]
    return Z[:, :k] if is_factorized else Z


# Phase 2: A_norm* evaluation for one (k, mult, seed_i) run

def _eval_one_run(run_dir: Path, k: int, mult: int, seed_i: int,
                  mc_list: list[ModelConfig]) -> list[dict]:
    """Load latents and compute A_norm* for every model in mc_list."""
    s = SEED + seed_i * 37
    n_train = mult * k
    records = []

    for mc in mc_list:
        lat_path = run_dir / mc.name / "latents.npz"
        if not lat_path.exists():
            print(f"  [SKIP] missing {lat_path}", flush=True)
            continue

        lats = np.load(lat_path)
        zh_ap = _std(_shared_rep(lats, "Zh_A_prime", k, mc.is_factorized))
        zr_ap = _std(_shared_rep(lats, "Zr_A_prime", k, mc.is_factorized))

        rng = np.random.default_rng(s + abs(hash(mc.name)) % (1 << 31))

        null_res = compute_anorm_null(
            zh_ap, zr_ap,
            n_permutations=N_PERMS,
            rng=np.random.default_rng(rng.integers(1 << 63)),
        )
        anorm_obs = null_res["anorm"]
        null_mean = float(np.mean(null_res["null"]))
        null_upper = null_res["null_upper"]

        denom = 1.0 - null_mean
        anorm_star = (anorm_obs - null_mean) / denom if abs(denom) > 1e-12 else 0.0

        # Bootstrap CI on A_norm*
        n = zh_ap.shape[0]
        boot_seeds = rng.integers(1 << 63, size=N_BOOT)
        boot_stars = []
        for bs in boot_seeds:
            idx = np.random.default_rng(bs).choice(n, size=n, replace=True)
            ba = compute_anorm(zh_ap[idx], zr_ap[idx])
            boot_stars.append((ba - null_mean) / denom if abs(denom) > 1e-12 else 0.0)
        boot_arr = np.array(boot_stars)

        records.append({
            "k": k,
            "n_train": n_train,
            "n_train_mult": mult,
            "seed_i": seed_i,
            "seed": s,
            "model": mc.name,
            "model_type": mc.model_type,
            "is_factorized": mc.is_factorized,
            "anorm": float(anorm_obs),
            "anorm_star": float(anorm_star),
            "null_mean": null_mean,
            "null_upper": float(null_upper),
            "ci_lower": float(np.percentile(boot_arr, 2.5)),
            "ci_upper": float(np.percentile(boot_arr, 97.5)),
            "anorm_exceeds_null": bool(anorm_obs > null_upper),
        })

    return records


def main() -> None:
    """Run the Pre Step A alignment saturation study.

    Trains all model configurations across k values and N_train multipliers,
    evaluates A_norm* from saved latents in parallel, and writes results to JSON.
    """
    parser = argparse.ArgumentParser(description="Pre Step A: alignment saturation study")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 50, 100])
    parser.add_argument("--n-train-multipliers", type=int, nargs="+",
                        default=[10, 20, 30, 40, 50])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default=f"{DEFAULT_RESULTS}/pre_step_a.json")
    parser.add_argument("--work-dir", type=str,
                        default=f"{DEFAULT_RESULTS}/pre_step_a_work")
    # Measurement regime betas — default is the locked baseline regime.
    # Pass different values to run Pre Step A for other regimes.
    parser.add_argument("--beta-s", type=float, default=1.0)
    parser.add_argument("--beta-h", type=float, default=1.0)
    parser.add_argument("--beta-r", type=float, default=1.0)
    parser.add_argument("--beta-b", type=float, default=0.75)
    args = parser.parse_args()

    k_values    = args.k
    multipliers = sorted(args.n_train_multipliers)
    n_seeds     = args.n_seeds
    work_dir    = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    beta_s, beta_h, beta_r, beta_b = args.beta_s, args.beta_h, args.beta_r, args.beta_b

    # Build the master list of (cfg, run_id, mc_list) for every combination.
    all_entries: list[tuple[SimulatorConfig, str, list[ModelConfig]]] = []
    for k in k_values:
        mc_list = model_configs_for_k(k)
        for mult in multipliers:
            n_train = mult * k
            for seed_i in range(n_seeds):
                s = SEED + seed_i * 37
                cfg = SimulatorConfig(
                    latent_dim=k,
                    d_h=D_OBS, d_r=D_OBS,
                    n_a=n_train,
                    n_a_prime=N_A_PRIME,
                    n_b=100, n_c=100,
                    beta_s=beta_s, beta_h=beta_h, beta_r=beta_r, beta_b=beta_b,
                    seed=s,
                )
                all_entries.append((cfg, _run_id(k, mult, seed_i), mc_list))

    n_models = len(model_configs_for_k(k_values[0]))
    n_runs   = len(all_entries)
    n_total  = n_runs * n_models
    n_batches = (n_runs + BATCH_SIZE - 1) // BATCH_SIZE

    print("Pre Step A: alignment saturation study")
    print(f"  k values:            {k_values}")
    print(f"  N_train multipliers: {multipliers}")
    print(f"  n_seeds:             {n_seeds}")
    print(f"  n_gpus:              8")
    print(f"  batch_size:          {BATCH_SIZE}")
    print(f"  d_obs:               {D_OBS}")
    print(f"  N_a_prime:           {N_A_PRIME}")
    print(f"  model configs/k:     {n_models}")
    print(f"  total runs:          {n_runs}  ({n_batches} batches)")
    print(f"  total trainings:     {n_total}")
    print(f"  work_dir:            {work_dir}")
    print(f"  output:              {args.output}")
    print(flush=True)

    t0 = time.time()

    # -----------------------------------------------------------------------
    # Phase 1: Train all runs (batched GPU pool, BATCH_SIZE=8)
    # -----------------------------------------------------------------------
    print("Phase 1: Training all runs (batched GPU pool)...", flush=True)
    t_phase = time.time()

    for batch_start in range(0, n_runs, BATCH_SIZE):
        batch = all_entries[batch_start:batch_start + BATCH_SIZE]
        b_num = batch_start // BATCH_SIZE + 1

        # Group by k within the batch so train_runs_gpu_pool gets a consistent
        # mc_list (mc_list depends only on k).
        by_k: dict[int, list[tuple[SimulatorConfig, str]]] = defaultdict(list)
        mc_for_k: dict[int, list[ModelConfig]] = {}
        for cfg, run_id, mc_list in batch:
            by_k[cfg.latent_dim].append((cfg, run_id))
            mc_for_k[cfg.latent_dim] = mc_list

        for k_batch, run_pairs in by_k.items():
            train_runs_gpu_pool(run_pairs, mc_for_k[k_batch], work_dir, n_gpus=8)

        print(f"  batch {b_num}/{n_batches} done  ({time.time()-t0:.0f}s elapsed)",
              flush=True)

    print(f"Phase 1 complete: {n_runs} runs in {time.time()-t_phase:.0f}s\n",
          flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: Evaluate A_norm* from saved latents (Parallel, n_jobs=32)
    # -----------------------------------------------------------------------
    print("Phase 2: Evaluating A_norm* (n_jobs=32)...", flush=True)
    t_phase = time.time()

    per_run = Parallel(n_jobs=32)(
        delayed(_eval_one_run)(
            work_dir / run_id, cfg.latent_dim,
            cfg.n_a // cfg.latent_dim,   # mult
            (cfg.seed - SEED) // 37,     # seed_i
            mc_list,
        )
        for cfg, run_id, mc_list in all_entries
    )

    all_records = [rec for run_recs in per_run for rec in run_recs]

    print(f"Phase 2 complete: {len(all_records)} records in {time.time()-t_phase:.0f}s\n",
          flush=True)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_records, f, indent=2)

    elapsed = time.time() - t0
    print(f"Done: {len(all_records)} records saved to {out}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Quick summary table
    import pandas as pd
    df = pd.DataFrame(all_records)
    print("\nA_norm* summary (mean across seeds):")
    summary = (
        df.groupby(["k", "n_train_mult", "model"])["anorm_star"]
        .mean()
        .unstack("model")
    )
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
