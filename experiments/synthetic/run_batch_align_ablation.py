"""Batch alignment ablation — tests S2B detection at different ρ_b values.

Runs the same experiment as the main baseline but with batch_align ∈ {0.0, 0.5}
instead of the default 1.0. Since Cohort A (training data) has b ~ N(0, I)
regardless of batch_align, the confounding mechanism is identical — only the
cohort shift direction between B and C changes.

At ρ_b=0: shift is random (orthogonal to confound) → D_quantile blind → S2 not detected
At ρ_b=0.5: shift partially aligned → reduced detection
At ρ_b=1.0: shift fully aligned (existing data) → full detection

Design: 50 seeds × 7 models × 2 new conditions. Evaluates S1 + S2B configs
(8 alpha configs) at N_eval=1000 only. Total: 700 trainings + eval.

Expected runtime: ~3h on 8 GPUs (training) + ~1h (eval).

Usage
-----
    python3 experiments/synthetic/run_batch_align_ablation.py
    python3 experiments/synthetic/run_batch_align_ablation.py --phase train
    python3 experiments/synthetic/run_batch_align_ablation.py --phase eval
    python3 experiments/synthetic/run_batch_align_ablation.py --n-runs-override 5  # smoke test
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.evaluate_scenarios import (
    AlphaConfig, EvalConfig, evaluate_runs_parallel, make_alpha_configs,
)
from pipeline.train_models import ModelConfig, is_run_trained, train_runs_gpu_pool
from simulator.simulator import SimulatorConfig

# Constants — match main experiment
K          = 50
N_TRAIN    = 1500
N_EVAL     = 1000
D_OBS      = 1024
BSM        = 2.0
N_SEEDS    = 50
SEED       = 42
SEED_STRIDE = 37

OUTPUT_BASE = Path(os.environ.get(
    "DECAT_BATCH_ALIGN_DIR",
    "./data/batch_align_ablation",
))
REPR_DIR    = OUTPUT_BASE / "repr"
RESULTS_DIR = OUTPUT_BASE / "results"

BASELINE_REGIME = {"beta_s": 1.0, "beta_h": 1.0, "beta_r": 1.0, "beta_b": 0.75}

# Test conditions: ρ_b = 0.0 (random shift) and 0.5 (partial alignment)
# ρ_b = 1.0 is the existing main experiment data (no re-run needed)
BATCH_ALIGN_VALUES = [0.0, 0.5]


def _model_configs(k: int = K) -> list[ModelConfig]:
    return [
        ModelConfig("cca",  f"cca_K{k}", {"n_components": k}, False),
        ModelConfig("clip", f"clip_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4, "temperature": 0.07,
            "lr": 1e-3, "n_epochs": 200, "batch_size": 1024,
        }, False),
        ModelConfig("jive", f"jive_K{k}", {"n_components": k, "n_individual": k}, True),
        ModelConfig("dssl", f"dssl_l0.01_b0.01_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4, "beta": 0.01, "lambda_orth": 0.01,
            "temperature": 0.07, "aug_noise_std": 0.1, "lr": 1e-3,
            "n_epochs_shared": 125, "n_epochs_specific": 50, "batch_size": 1024,
        }, True),
        ModelConfig("dssl", f"dssl_l0.1_b0.1_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4, "beta": 0.1, "lambda_orth": 0.1,
            "temperature": 0.07, "aug_noise_std": 0.1, "lr": 1e-3,
            "n_epochs_shared": 125, "n_epochs_specific": 50, "batch_size": 1024,
        }, True),
        ModelConfig("dssl", f"dssl_l1.0_b1.0_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4, "beta": 1.0, "lambda_orth": 1.0,
            "temperature": 0.07, "aug_noise_std": 0.1, "lr": 1e-3,
            "n_epochs_shared": 75, "n_epochs_specific": 50, "batch_size": 1024,
        }, True),
        ModelConfig("dssl", f"dssl_l10.0_b10.0_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4, "beta": 10.0, "lambda_orth": 10.0,
            "temperature": 0.07, "aug_noise_std": 0.1, "lr": 1e-3,
            "n_epochs_shared": 75, "n_epochs_specific": 50, "batch_size": 1024,
        }, True),
    ]


def _make_cfg(seed: int, batch_align: float) -> SimulatorConfig:
    """Build a SimulatorConfig with specified batch_align."""
    return SimulatorConfig(
        latent_dim=K, d_h=D_OBS, d_r=D_OBS,
        n_a=N_TRAIN, n_a_prime=N_EVAL, n_b=N_EVAL, n_c=N_EVAL,
        alpha_s=0.0, alpha_h=0.0, alpha_r=0.0, alpha_b=0.0,
        batch_align=batch_align,
        batch_shift_magnitude=BSM,
        seed=seed,
        **BASELINE_REGIME,
    )


def build_run_configs(
    n_runs_override: int | None = None,
) -> list[tuple[SimulatorConfig, str, float]]:
    """Return (cfg, run_id, batch_align) for all runs."""
    configs = []
    for ba in BATCH_ALIGN_VALUES:
        ba_tag = f"ba{ba:.1f}".replace(".", "")
        n_seeds = N_SEEDS if n_runs_override is None else min(N_SEEDS, n_runs_override)
        for i in range(n_seeds):
            seed = SEED + i * SEED_STRIDE
            run_id = f"{ba_tag}_baseline_{i:03d}"
            cfg = _make_cfg(seed, ba)
            configs.append((cfg, run_id, ba))
    return configs


def _s2b_s1_alpha_configs() -> list[AlphaConfig]:
    """S1 + S2B alpha configs only (8 total — the comparison set)."""
    all_cfgs = make_alpha_configs()
    return [c for c in all_cfgs
            if c.config_id.startswith(('s1_', 's2b_'))]


def train_phase(run_configs, repr_dir, batch_size=8, n_gpus=None, skip_existing=True):
    if n_gpus is None:
        import torch
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    repr_dir.mkdir(parents=True, exist_ok=True)
    model_configs = _model_configs()
    pending = [
        (cfg, run_id) for cfg, run_id, _ in run_configs
        if not skip_existing or not is_run_trained(repr_dir / run_id, model_configs)
    ]
    n_skip = len(run_configs) - len(pending)
    print(f"\nTraining: {len(pending)} runs to train, {n_skip} skipped")
    if not pending:
        return
    t0 = time.time()
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        n_batches = (len(pending) + batch_size - 1) // batch_size
        elapsed = time.time() - t0
        print(f"  batch {batch_num}/{n_batches}  ({elapsed:.0f}s elapsed)", flush=True)
        train_runs_gpu_pool(batch, model_configs, repr_dir, n_gpus=n_gpus)
    print(f"Training complete in {time.time()-t0:.0f}s", flush=True)


def eval_phase(run_configs, repr_dir, results_dir, n_concurrent=100):
    results_dir.mkdir(parents=True, exist_ok=True)
    alpha_cfgs = _s2b_s1_alpha_configs()
    eval_cfg = EvalConfig(
        n_eval_sizes=[N_EVAL],
        n_bootstrap=200,
        n_permutations=200,
        n_jobs=1,
    )
    run_ids = [run_id for _, run_id, _ in run_configs]
    print(f"\nEval: {len(run_ids)} runs × {len(alpha_cfgs)} alpha configs", flush=True)
    evaluate_runs_parallel(
        run_ids=run_ids,
        repr_dir=repr_dir,
        alpha_configs=alpha_cfgs,
        eval_config=eval_cfg,
        output_dir=results_dir,
        n_concurrent=n_concurrent,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch alignment ablation (ρ_b sweep)")
    parser.add_argument("--phase", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--n-runs-override", type=int, default=None)
    parser.add_argument("--n-concurrent", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    run_configs = build_run_configs(args.n_runs_override)
    total_trainings = len(run_configs) * 7

    print("=" * 65)
    print("DECAT Batch Alignment Ablation")
    print(f"  Conditions:      ρ_b ∈ {BATCH_ALIGN_VALUES}")
    print(f"  Seeds/condition: {N_SEEDS if args.n_runs_override is None else args.n_runs_override}")
    print(f"  Total runs:      {len(run_configs)}")
    print(f"  Model trainings: {total_trainings}")
    print(f"  Alpha configs:   S1 (4) + S2B (4) = 8")
    print(f"  Phase:           {args.phase}")
    print(f"  repr_dir:        {REPR_DIR}")
    print(f"  results_dir:     {RESULTS_DIR}")
    print("=" * 65, flush=True)

    t_start = time.time()

    if args.phase in ("train", "all"):
        train_phase(run_configs, REPR_DIR, skip_existing=args.skip_existing)

    if args.phase in ("eval", "all"):
        eval_phase(run_configs, REPR_DIR, RESULTS_DIR, n_concurrent=args.n_concurrent)

    print(f"\nTotal elapsed: {(time.time()-t_start)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
