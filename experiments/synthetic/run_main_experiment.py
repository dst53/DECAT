"""Main experiment — Step A (training) + Step B (evaluation).

Generates and trains representations for all 365 runs across the full
measurement regime × proxy configuration grid, then evaluates each run
against the predefined alpha configurations.

Run counts
----------
- 125 baseline runs (γ_h=γ_r=0):
    50 baseline regime + 25 per 3 other measurement regimes
- 240 proxy-entangled runs (8 proxy configs × 30 runs each):
    each proxy config split across 4 measurement regimes (8/8/7/7)

Total: 365 runs × 7 models = 2,555 model trainings.

Evaluation alpha configs
------------------------
- Baseline runs: make_alpha_configs() → 28 configs
- Proxy runs: 4 S1 configs + make_proxy_alpha_configs(gamma, eta) → 21 configs

Eval wall time: ~303K calls × 5s avg / n_concurrent=100 ≈ 8h
  (n_bootstrap=200, n_permutations=200; same as gate calibration)

Usage
-----
    # Full run (train then eval):
    python3 experiments/run_main_experiment.py

    # Train only (then eval separately):
    python3 experiments/run_main_experiment.py --phase train

    # Eval only (representations already exist):
    python3 experiments/run_main_experiment.py --phase eval

    # Quick smoke test (2 runs per group):
    python3 experiments/run_main_experiment.py --n-runs-override 2
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.evaluate_scenarios import (
    EvalConfig, evaluate_runs_parallel,
    make_alpha_configs, make_proxy_alpha_configs,
)
from pipeline.train_models import ModelConfig, is_run_trained, train_runs_gpu_pool
from simulator.simulator import SimulatorConfig

# Constants

K          = 50
N_TRAIN    = 1500       # 30×k — validated by Pre Step A across all regimes
N_EVAL     = 1000       # n_a_prime / n_b / n_c
D_OBS      = 1024
BSM        = 2.0        # batch_shift_magnitude — locked operating point
SEED       = 42
SEED_STRIDE = 37

OUTPUT_BASE = Path(os.environ.get("DECAT_RESULTS_DIR", "./data/main_experiment"))
REPR_DIR    = OUTPUT_BASE / "repr"
RESULTS_DIR = OUTPUT_BASE / "results"

# Measurement regimes (spec §05 §8)
# Measurement regimes — each non-baseline regime doubles exactly one coefficient
# from its baseline value (2× β_s, 2× β_b=0.75→1.5, 2× β_h/β_r).
# beta_b=0.75 baseline is the locked S2B operating point from gate calibration.

MEASUREMENT_REGIMES = {
    "baseline":           {"beta_s": 1.0, "beta_h": 1.0, "beta_r": 1.0, "beta_b": 0.75},
    "shared_dominant":    {"beta_s": 2.0, "beta_h": 1.0, "beta_r": 1.0, "beta_b": 0.75},
    "batch_dominant":     {"beta_s": 1.0, "beta_h": 1.0, "beta_r": 1.0, "beta_b": 1.5},
    "modality_dominant":  {"beta_s": 1.0, "beta_h": 2.0, "beta_r": 2.0, "beta_b": 0.75},
}

# Runs per regime for baseline group (total = 125)
BASELINE_RUNS_PER_REGIME = {
    "baseline":           50,
    "shared_dominant":    25,
    "batch_dominant":     25,
    "modality_dominant":  25,
}

# Proxy configurations (spec §05 §8 proxy parameter grid, 8 configs)

PROXY_CONFIGS = [
    {"name": "g0.3_r0_e0.3",  "gamma_h": 0.3, "gamma_r": 0.0, "eta": 0.3},
    {"name": "g0.5_r0_e0",    "gamma_h": 0.5, "gamma_r": 0.0, "eta": 0.0},
    {"name": "g0.5_r0_e0.3",  "gamma_h": 0.5, "gamma_r": 0.0, "eta": 0.3},
    {"name": "g0.5_r0_e0.6",  "gamma_h": 0.5, "gamma_r": 0.0, "eta": 0.6},
    {"name": "g1.0_r0_e0",    "gamma_h": 1.0, "gamma_r": 0.0, "eta": 0.0},
    {"name": "g1.0_r0_e0.3",  "gamma_h": 1.0, "gamma_r": 0.0, "eta": 0.3},
    {"name": "g0.5_r0.5_e0",  "gamma_h": 0.5, "gamma_r": 0.5, "eta": 0.0},
    {"name": "g1.0_r1.0_e0",  "gamma_h": 1.0, "gamma_r": 1.0, "eta": 0.0},
]

# Runs per regime for each proxy config (must sum to 30)
# baseline and shared_dominant get 8; batch_dominant and modality_dominant get 7
PROXY_RUNS_PER_REGIME = {
    "baseline":           8,
    "shared_dominant":    8,
    "batch_dominant":     7,
    "modality_dominant":  7,
}
assert sum(PROXY_RUNS_PER_REGIME.values()) == 30

# Model configs (7 models — same as gate calibration)

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


# SimulatorConfig builder

def _make_cfg(seed: int, regime_name: str,
              gamma_h: float = 0.0, gamma_r: float = 0.0, eta: float = 0.0,
              proxy_shift_align: float = 0.0) -> SimulatorConfig:
    """Build a SimulatorConfig for one run.

    proxy_shift_align=1.0 for proxy runs so the cohort shift is aligned with
    the proxy direction, enabling dq_task to detect proxy S2 signal.
    """
    regime_betas = MEASUREMENT_REGIMES[regime_name]
    return SimulatorConfig(
        latent_dim=K, d_h=D_OBS, d_r=D_OBS,
        n_a=N_TRAIN, n_a_prime=N_EVAL, n_b=N_EVAL, n_c=N_EVAL,
        alpha_s=0.0, alpha_h=0.0, alpha_r=0.0, alpha_b=0.0,
        gamma_h=gamma_h, gamma_r=gamma_r, eta=eta,
        batch_align=1.0,
        proxy_shift_align=proxy_shift_align,
        proxy_shift_target="h",
        batch_shift_magnitude=BSM,
        seed=seed,
        **regime_betas,
    )


# Run config builders

def build_baseline_run_configs(
    n_runs_override: int | None = None,
) -> list[tuple[SimulatorConfig, str, str]]:
    """Return (cfg, run_id, regime_name) for all 125 baseline runs."""
    configs = []
    for regime_name, n_runs in BASELINE_RUNS_PER_REGIME.items():
        if n_runs_override is not None:
            n_runs = min(n_runs, n_runs_override)
        for i in range(n_runs):
            seed = SEED + (len(configs) + i) * SEED_STRIDE
            run_id = f"baseline_{regime_name}_{i:03d}"
            cfg = _make_cfg(seed, regime_name)
            configs.append((cfg, run_id, regime_name))
    return configs


def build_proxy_run_configs(
    n_runs_override: int | None = None,
) -> list[tuple[SimulatorConfig, str, str, dict]]:
    """Return (cfg, run_id, regime_name, proxy_cfg) for all 240 proxy runs."""
    configs = []
    for proxy_cfg in PROXY_CONFIGS:
        pname = proxy_cfg["name"]
        gh, gr, eta = proxy_cfg["gamma_h"], proxy_cfg["gamma_r"], proxy_cfg["eta"]
        for regime_name, n_runs in PROXY_RUNS_PER_REGIME.items():
            if n_runs_override is not None:
                n_runs = min(n_runs, n_runs_override)
            for i in range(n_runs):
                seed = SEED + (len(configs) + i) * SEED_STRIDE
                run_id = f"proxy_{pname}_{regime_name}_{i:03d}"
                cfg = _make_cfg(seed, regime_name,
                                gamma_h=gh, gamma_r=gr, eta=eta,
                                proxy_shift_align=1.0)
                configs.append((cfg, run_id, regime_name, proxy_cfg))
    return configs


# Training phase

def train_phase(
    run_configs: list,
    repr_dir: Path,
    batch_size: int = 8,
    n_gpus: int = 8,
    skip_existing: bool = True,
) -> None:
    """Train all 7 models for every run in run_configs.

    Parameters
    ----------
    run_configs : list
        List of (SimulatorConfig, run_id, ...) tuples from build_baseline_run_configs
        or build_proxy_run_configs.
    repr_dir : Path
        Directory where per-run representation subdirectories will be written.
    batch_size : int
        Number of runs dispatched to the GPU pool per batch.
    n_gpus : int
        Number of GPUs available for the pool.
    skip_existing : bool
        If True, skip runs where every model has finished training.
    """
    repr_dir.mkdir(parents=True, exist_ok=True)
    model_configs = _model_configs()

    # Filter already-trained runs
    pending = []
    for entry in run_configs:
        cfg, run_id = entry[0], entry[1]
        run_dir = repr_dir / run_id
        if skip_existing and is_run_trained(run_dir, model_configs):
            continue
        pending.append((cfg, run_id))

    n_total = len(pending)
    n_skip  = len(run_configs) - n_total
    print(f"\nTraining: {n_total} runs to train, {n_skip} skipped (already done)")
    if n_total == 0:
        return

    t0 = time.time()
    for batch_start in range(0, n_total, batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        n_batches = (n_total + batch_size - 1) // batch_size
        elapsed = time.time() - t0
        rate = batch_num / max(elapsed, 1)
        eta = (n_batches - batch_num) / rate if rate > 0 else float("inf")
        print(f"  batch {batch_num}/{n_batches}  "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)
        train_runs_gpu_pool(batch, model_configs, repr_dir, n_gpus=n_gpus)

    print(f"Training complete in {time.time()-t0:.0f}s", flush=True)


# Evaluation phase

def _s1_alpha_configs():
    """Return only the 4 S1 configs from make_alpha_configs()."""
    cfgs = [c for c in make_alpha_configs() if c.alpha_s > 0 and
            c.alpha_b == 0 and c.alpha_h == 0 and c.alpha_r == 0]
    assert len(cfgs) == 4, f"Expected 4 S1 configs, got {len(cfgs)}"
    return cfgs


def eval_phase_baseline(
    run_configs: list,
    repr_dir: Path,
    results_dir: Path,
    n_concurrent: int = 100,
    skip_existing: bool = True,
) -> None:
    """Evaluate all baseline runs with make_alpha_configs() (28 configs)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    alpha_cfgs = make_alpha_configs()
    eval_cfg = EvalConfig(
        n_eval_sizes=[50, 100, 200, 500, 1000],
        n_bootstrap=200,
        n_permutations=200,
        n_jobs=1,
    )

    run_ids = [entry[1] for entry in run_configs]
    print(f"\nEval (baseline): {len(run_ids)} runs × {len(alpha_cfgs)} alpha configs",
          flush=True)
    evaluate_runs_parallel(
        run_ids=run_ids,
        repr_dir=repr_dir,
        alpha_configs=alpha_cfgs,
        eval_config=eval_cfg,
        output_dir=results_dir,
        n_concurrent=n_concurrent,
    )


def eval_phase_proxy(
    run_configs: list,
    repr_dir: Path,
    results_dir: Path,
    n_concurrent: int = 100,
    skip_existing: bool = True,
) -> None:
    """Evaluate proxy runs grouped by (gamma_h, gamma_r, eta).

    Each group uses: 4 S1 configs + make_proxy_alpha_configs(gamma, eta) = 21 total.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = EvalConfig(
        n_eval_sizes=[50, 100, 200, 500, 1000],
        n_bootstrap=200,
        n_permutations=200,
        n_jobs=1,
    )
    s1_cfgs = _s1_alpha_configs()

    # Group by proxy config (gamma_h, gamma_r, eta)
    groups: dict[tuple, list] = defaultdict(list)
    for entry in run_configs:
        _, run_id, _, proxy_cfg = entry
        key = (proxy_cfg["gamma_h"], proxy_cfg["gamma_r"], proxy_cfg["eta"])
        groups[key].append(run_id)

    for (gh, gr, eta), run_ids in sorted(groups.items()):
        proxy_cfgs = make_proxy_alpha_configs(gamma=gh, eta=eta)
        alpha_cfgs = s1_cfgs + proxy_cfgs
        print(f"\nEval (proxy γ_h={gh}, γ_r={gr}, η={eta}): "
              f"{len(run_ids)} runs × {len(alpha_cfgs)} alpha configs", flush=True)
        evaluate_runs_parallel(
            run_ids=run_ids,
            repr_dir=repr_dir,
            alpha_configs=alpha_cfgs,
            eval_config=eval_cfg,
            output_dir=results_dir,
            n_concurrent=n_concurrent,
        )


# Entry point

def main() -> None:
    """Parse CLI args and run the main experiment (train, eval, or both)."""
    parser = argparse.ArgumentParser(description="DECAT main experiment")
    parser.add_argument("--phase", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--proxy-only",    action="store_true")
    parser.add_argument("--n-runs-override", type=int, default=None,
                        help="Cap runs per group (smoke test)")
    parser.add_argument("--n-concurrent",  type=int, default=100)
    parser.add_argument("--repr-dir",    type=str, default=str(REPR_DIR))
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    repr_dir    = Path(args.repr_dir)
    results_dir = Path(args.results_dir)

    # Build run config lists
    baseline_runs = ([] if args.proxy_only
                     else build_baseline_run_configs(args.n_runs_override))
    proxy_runs    = ([] if args.baseline_only
                     else build_proxy_run_configs(args.n_runs_override))

    total_runs = len(baseline_runs) + len(proxy_runs)
    total_models = total_runs * 7

    print("=" * 65)
    print("DECAT Main Experiment")
    print(f"  Baseline runs:   {len(baseline_runs):>5}  (125 planned)")
    print(f"  Proxy runs:      {len(proxy_runs):>5}  (240 planned)")
    print(f"  Total runs:      {total_runs:>5}")
    print(f"  Model trainings: {total_models:>5}  (7 per run)")
    print(f"  Phase:           {args.phase}")
    print(f"  repr_dir:        {repr_dir}")
    print(f"  results_dir:     {results_dir}")
    print("=" * 65, flush=True)

    t_start = time.time()

    if args.phase in ("train", "all"):
        if baseline_runs:
            print("\n--- Training: baseline runs ---", flush=True)
            train_phase(baseline_runs, repr_dir,
                        skip_existing=args.skip_existing)
        if proxy_runs:
            print("\n--- Training: proxy runs ---", flush=True)
            train_phase(proxy_runs, repr_dir,
                        skip_existing=args.skip_existing)

    if args.phase in ("eval", "all"):
        if baseline_runs:
            print("\n--- Evaluation: baseline runs ---", flush=True)
            eval_phase_baseline(baseline_runs, repr_dir, results_dir,
                                n_concurrent=args.n_concurrent,
                                skip_existing=args.skip_existing)
        if proxy_runs:
            print("\n--- Evaluation: proxy runs ---", flush=True)
            eval_phase_proxy(proxy_runs, repr_dir, results_dir,
                             n_concurrent=args.n_concurrent,
                             skip_existing=args.skip_existing)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
