"""Pre Step A.2 — Training convergence validation.

Verifies that CLIP and DSSL epoch counts are in the converged regime at
the N_train selected by Pre Step A (N_train=1500, k=50).

For CLIP and DSSL, trains with checkpoint callbacks every
CHECKPOINT_INTERVAL epochs and records A_norm* on held-out Cohort A'.
JIVE is closed-form (no training dynamics) — a single A_norm* measurement
is recorded.

Parallelism: 5 seeds are run concurrently via ProcessPoolExecutor with
max_workers=5 (one GPU per worker).
Each worker process handles one seed: trains all models sequentially
(checkpoints are synchronous within each training) and stores records.

DSSL configs: (0.01,0.01), (0.1,0.1), (1.0,1.0), (10.0,10.0) — all four
diagonal configurations.

Output: JSON array of records saved to
  ./results/pre_step_a2.json

Usage:
  python experiments/run_pre_step_a2.py
  python experiments/run_pre_step_a2.py --n-seeds 1 --max-epoch-mult 1.5
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

DEFAULT_RESULTS = os.environ.get("DECAT_RESULTS_DIR", "./results")

SEED               = 42
K                  = 50
D_OBS              = 1024
N_TRAIN            = 1500   # 30 × K from Pre Step A
N_A_PRIME          = 500
N_PERMS            = 200
N_BOOT             = 200
CHECKPOINT_INTERVAL = 25

# DSSL configs: all four diagonal (β = λ) points
DSSL_CONFIGS = [
    ("DSSL (0.01)", {"beta": 0.01, "lambda_orth": 0.01,
                     "n_epochs_shared": 125, "n_epochs_specific": 50}),
    ("DSSL (0.1)",  {"beta": 0.1,  "lambda_orth": 0.1,
                     "n_epochs_shared": 125, "n_epochs_specific": 50}),
    ("DSSL (1.0)",  {"beta": 1.0,  "lambda_orth": 1.0,
                     "n_epochs_shared": 75,  "n_epochs_specific": 50}),
    ("DSSL (10.0)", {"beta": 10.0, "lambda_orth": 10.0,
                     "n_epochs_shared": 75,  "n_epochs_specific": 50}),
]


# Shared helpers (must be importable in worker subprocesses)

def _std(Z: np.ndarray) -> np.ndarray:
    return (Z - Z.mean(0)) / Z.std(0).clip(min=1e-8)


def _compute_anorm_star(
    zh_ap: np.ndarray,
    zr_ap: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Compute A_norm* with permutation null and bootstrap CI."""
    from metrics.anorm import compute_anorm, compute_anorm_null

    null_res  = compute_anorm_null(
        zh_ap, zr_ap,
        n_permutations=N_PERMS,
        rng=np.random.default_rng(rng.integers(1 << 63)),
    )
    anorm_obs = null_res["anorm"]
    null_mean = float(np.mean(null_res["null"]))
    denom     = 1.0 - null_mean
    anorm_star = (anorm_obs - null_mean) / denom if abs(denom) > 1e-12 else 0.0

    n = zh_ap.shape[0]
    boot_seeds = rng.integers(1 << 63, size=N_BOOT)
    boot_stars = []
    for bs in boot_seeds:
        idx = np.random.default_rng(bs).choice(n, size=n, replace=True)
        ba = compute_anorm(zh_ap[idx], zr_ap[idx])
        boot_stars.append((ba - null_mean) / denom if abs(denom) > 1e-12 else 0.0)
    boot_arr = np.array(boot_stars)

    return {
        "anorm":      float(anorm_obs),
        "anorm_star": float(anorm_star),
        "null_mean":  null_mean,
        "ci_lower":   float(np.percentile(boot_arr, 2.5)),
        "ci_upper":   float(np.percentile(boot_arr, 97.5)),
    }


# Per-seed worker

def _run_seed(seed_i: int, gpu_id: int,
              clip_max_epochs: int, epoch_mult: float,
              beta_s: float = 1.0, beta_h: float = 1.0,
              beta_r: float = 1.0, beta_b: float = 0.75) -> list[dict]:
    """Train all models for one seed on the assigned GPU and return records.

    Called in a subprocess. Returns a list of record dicts.
    ``epoch_mult`` is applied to each DSSL config's default epoch counts.
    """
    import torch
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from simulator.simulator import SimulatorConfig, simulate_run
    from models.clip import CLIP
    from models.disentangled_ssl import DisentangledSSL


    s = SEED + seed_i * 37
    rng = np.random.default_rng(s)

    print(f"\n[seed {seed_i} GPU {gpu_id}] Generating data (N_train={N_TRAIN})...",
          flush=True)
    sim_cfg = SimulatorConfig(
        latent_dim=K, d_h=D_OBS, d_r=D_OBS,
        n_a=N_TRAIN, n_a_prime=N_A_PRIME,
        n_b=100, n_c=100,
        beta_s=beta_s, beta_h=beta_h, beta_r=beta_r, beta_b=beta_b,
        seed=s,
    )
    data    = simulate_run(sim_cfg)
    mask_a  = data["cohort"] == "A"
    mask_ap = data["cohort"] == "A_prime"
    xh_a  = data["xh"][mask_a]
    xr_a  = data["xr"][mask_a]
    xh_ap = data["xh"][mask_ap]
    xr_ap = data["xr"][mask_ap]

    records: list[dict] = []
    device = f"cuda:{gpu_id}"

    # Helper: build a record dict and log it
    def _record(model_name: str, phase: str, epoch: int, epoch_loss: float,
                zh: np.ndarray, zr: np.ndarray) -> None:
        astar = _compute_anorm_star(
            _std(zh), _std(zr),
            rng=np.random.default_rng(rng.integers(1 << 63)),
        )
        rec = {
            "seed_i": seed_i, "seed": s,
            "model": model_name, "phase": phase,
            "epoch": epoch, "train_loss": float(epoch_loss),
            **astar,
        }
        records.append(rec)
        print(
            f"  [seed {seed_i} GPU {gpu_id}] {model_name:<14s} {phase:<8s} "
            f"ep={epoch:>4d}  "
            f"A*={astar['anorm_star']:.3f} "
            f"[{astar['ci_lower']:.3f},{astar['ci_upper']:.3f}]  "
            f"loss={epoch_loss:.4f}",
            flush=True,
        )

    # CLIP
    print(f"[seed {seed_i} GPU {gpu_id}] Training CLIP ({clip_max_epochs} epochs)...",
          flush=True)

    def clip_cb(epoch: int, model: CLIP, epoch_loss: float) -> None:
        with torch.no_grad():
            dev = torch.device(model.device)
            zh = model._enc_h(
                torch.tensor(xh_ap, dtype=torch.float32, device=dev)
            ).cpu().numpy()
            zr = model._enc_r(
                torch.tensor(xr_ap, dtype=torch.float32, device=dev)
            ).cpu().numpy()
        _record("CLIP", "main", epoch, epoch_loss, zh, zr)

    CLIP(
        latent_dim=K, hidden_multiplier=4, temperature=0.07,
        lr=1e-3, n_epochs=clip_max_epochs, batch_size=1024, seed=s,
        device=device,
    ).fit(xh_a, xr_a, checkpoint_fn=clip_cb,
          checkpoint_interval=CHECKPOINT_INTERVAL)

    # DSSL — all four diagonal configs
    for dssl_name, dssl_kw in DSSL_CONFIGS:
        # Scale each config's default epoch counts by epoch_mult independently.
        n_shared   = int(dssl_kw["n_epochs_shared"]   * epoch_mult)
        n_specific = int(dssl_kw["n_epochs_specific"]  * epoch_mult)
        print(
            f"[seed {seed_i} GPU {gpu_id}] Training {dssl_name} "
            f"({n_shared}+{n_specific} epochs)...",
            flush=True,
        )

        def _make_dssl_cb(name: str):
            def callback(phase: str, epoch: int,
                         model: DisentangledSSL, epoch_loss: float) -> None:
                import torch as _torch
                dev = _torch.device(model.device)
                with _torch.no_grad():
                    if phase == "shared":
                        zh = model._enc_h_c(
                            _torch.tensor(xh_ap, dtype=_torch.float32, device=dev)
                        ).cpu().numpy()
                        zr = model._enc_r_c(
                            _torch.tensor(xr_ap, dtype=_torch.float32, device=dev)
                        ).cpu().numpy()
                    else:
                        zh = model.encode_h(xh_ap)[:, :K]
                        zr = model.encode_r(xr_ap)[:, :K]
                _record(name, phase, epoch, epoch_loss, zh, zr)
            return callback

        DisentangledSSL(
            latent_dim=K, hidden_multiplier=4,
            beta=dssl_kw["beta"], lambda_orth=dssl_kw["lambda_orth"],
            temperature=0.07, aug_noise_std=0.1, lr=1e-3,
            n_epochs_shared=n_shared, n_epochs_specific=n_specific,
            batch_size=1024, seed=s, device=device,
        ).fit(xh_a, xr_a,
              checkpoint_fn=_make_dssl_cb(dssl_name),
              checkpoint_interval=CHECKPOINT_INTERVAL)

        # Release GPU memory between DSSL configs
        torch.cuda.empty_cache()

    return records


def main() -> None:
    """Run the Pre Step A.2 training convergence validation.

    Trains CLIP and all four DSSL configurations to 2x their default epoch
    counts across multiple seeds, recording A_norm* at checkpoints. Results
    are written to JSON for analysis in notebook 12.
    """
    parser = argparse.ArgumentParser(description="Pre Step A.2: convergence validation")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--max-epoch-mult", type=float, default=2.0,
        help="Train for this multiple of default epochs (default: 2x)",
    )
    parser.add_argument(
        "--output", type=str,
        default=f"{DEFAULT_RESULTS}/pre_step_a2.json",
    )
    parser.add_argument("--beta-s", type=float, default=1.0)
    parser.add_argument("--beta-h", type=float, default=1.0)
    parser.add_argument("--beta-r", type=float, default=1.0)
    parser.add_argument("--beta-b", type=float, default=0.75)
    args = parser.parse_args()

    n_seeds = args.n_seeds
    mult    = args.max_epoch_mult

    clip_default_epochs = 200
    clip_max_epochs     = int(clip_default_epochs * mult)

    print("Pre Step A.2: training convergence validation")
    print(f"  regime: beta_s={args.beta_s} beta_h={args.beta_h} beta_r={args.beta_r} beta_b={args.beta_b}")
    print(f"  k={K}, d_obs={D_OBS}, N_train={N_TRAIN}, N_a_prime={N_A_PRIME}")
    print(f"  n_seeds:            {n_seeds}")
    print(f"  max_epoch_mult:     {mult}x")
    print(f"  CLIP epochs:        0..{clip_max_epochs} (default={clip_default_epochs})")
    for dssl_name, dssl_kw in DSSL_CONFIGS:
        print(f"  {dssl_name} shared: 0..{int(dssl_kw['n_epochs_shared']*mult)}"
              f"  specific: 0..{int(dssl_kw['n_epochs_specific']*mult)}")
    print(f"  checkpoint every:   {CHECKPOINT_INTERVAL} epochs")
    print(f"  parallelism:        {min(n_seeds, 5)} seeds × 1 GPU each")
    print(f"  output:             {args.output}")
    print(flush=True)

    t0 = time.time()
    all_records: list[dict] = []

    # Run seeds in parallel: up to 5 concurrent workers, each pinned to one GPU.
    max_workers = min(n_seeds, 5)
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=__import__("multiprocessing").get_context("spawn")) as exe:
        for seed_i in range(n_seeds):
            gpu_id = seed_i % max_workers   # 0..4, one GPU per worker
            fut = exe.submit(
                _run_seed, seed_i, gpu_id,
                clip_max_epochs, mult,
                args.beta_s, args.beta_h, args.beta_r, args.beta_b,
            )
            futures[fut] = seed_i

        for fut in as_completed(futures):
            seed_i = futures[fut]
            try:
                records = fut.result()
                all_records.extend(records)
                print(
                    f"\n[seed {seed_i}] complete — {len(records)} records  "
                    f"(total so far: {len(all_records)}, "
                    f"elapsed: {time.time()-t0:.0f}s)",
                    flush=True,
                )
            except Exception as exc:
                print(f"\n[seed {seed_i}] FAILED: {exc}", flush=True)
                raise

    # Save results
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_records, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_records)} records saved to {out}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
