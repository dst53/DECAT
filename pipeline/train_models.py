"""Representation learning pipeline for DECAT experiments.

For each synthetic run, trains all configured models on Cohort A and saves
learned representations for all four cohorts to disk.

Output structure::

    <output_dir>/<run_id>/
        metadata.json        -- SimulatorConfig and run parameters
        sim_data.npz         -- latents (z_s, z_h, z_r, b), outcome weights,
                                cohort labels; used by evaluate_scenarios.py
        <model_name>/
            latents.npz      -- Zh_{A,A_prime,B,C}, Zr_{A,A_prime,B,C}
            meta.json        -- model type, hyperparams, is_factorized flag
            gamma.npy        -- (JIVE, DSSL) sharedness scores; binary for JIVE
"""

import argparse
import json
import multiprocessing as mp
import os
import numpy as np
import joblib
import torch
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_RESULTS = os.environ.get("DECAT_RESULTS_DIR", "./results")

from simulator.simulator import SimulatorConfig, simulate_run
from models.cca import CCA
from models.jive import JIVE
from models.clip import CLIP
from models.disentangled_ssl import DisentangledSSL
from metrics.delta_shared import compute_sharedness_scores

COHORTS = ("A", "A_prime", "B", "C")


@dataclass
class ModelConfig:
    """Configuration for one representation model.

    Attributes
    ----------
    model_type : str
        One of ``"cca"``, ``"jive"``, ``"clip"``, or ``"dssl"``.
    name : str
        Unique identifier used as the output subdirectory name
        (e.g. ``"cca_K50"``, ``"mofa_K200"``).
    hyperparams : dict
        Keyword arguments passed directly to the model constructor.
    is_factorized : bool
        Whether the model exposes a shared/specific factorization.
    """

    model_type: str
    name: str
    hyperparams: dict
    is_factorized: bool


def default_model_configs(k: int) -> list[ModelConfig]:
    """Return the standard set of model configurations for latent dimensionality k.

    Includes all four model classes:
    - CCA: 1 config (top-k canonical components)
    - JIVE: 1 config (top-k components, linear factorized)
    - CLIP: 1 config (default hyperparameters)
    - DisentangledSSL: 2 configs bracketing the (lambda, beta) grid

    Parameters
    ----------
    k : int
        Ground-truth latent dimensionality; also the CCA / DSSL latent dim.

    Returns
    -------
    list[ModelConfig]
    """
    return [
        ModelConfig("cca",  f"cca_K{k}",  {"n_components": k}, False),
        ModelConfig("jive", f"jive_K{k}", {"n_components": k}, True),
        ModelConfig("clip", f"clip_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4, "temperature": 0.07,
            "lr": 1e-3, "n_epochs": 200, "batch_size": 1024,
        }, False),
        ModelConfig("dssl", f"dssl_l0.1_b0.1_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4,
            "beta": 0.1, "lambda_orth": 0.1,
            "temperature": 0.07, "aug_noise_std": 0.1,
            "lr": 1e-3, "n_epochs_shared": 125, "n_epochs_specific": 50,
            "batch_size": 1024,
        }, True),
        ModelConfig("dssl", f"dssl_l1.0_b1.0_K{k}", {
            "latent_dim": k, "hidden_multiplier": 4,
            "beta": 1.0, "lambda_orth": 1.0,
            "temperature": 0.07, "aug_noise_std": 0.1,
            "lr": 1e-3, "n_epochs_shared": 75, "n_epochs_specific": 50,
            "batch_size": 1024,
        }, True),
    ]


def train_run(
    sim_config: SimulatorConfig,
    model_configs: list[ModelConfig],
    output_dir: str | Path,
    run_id: str,
) -> None:
    """Train all models for one synthetic run and save representations to disk.

    Parameters
    ----------
    sim_config : SimulatorConfig
        Simulator parameters defining the data-generating process for this run.
        Outcome coefficients (alpha_*) should be zero since representation
        learning is outcome-agnostic.
    model_configs : list[ModelConfig]
        Models to train.
    output_dir : str or Path
        Root directory for outputs. A subdirectory named run_id is created.
    run_id : str
        Unique identifier for this run, used as the output subdirectory name.
    """
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{run_id}] Generating synthetic data...", flush=True)
    data = simulate_run(sim_config)

    cohort_masks = {c: data["cohort"] == c for c in COHORTS}

    with open(run_dir / "metadata.json", "w") as f:
        json.dump({"run_id": run_id, "sim_config": asdict(sim_config)}, f, indent=2)

    # Save latents and weights for outcome generation in evaluate_scenarios.py.
    # Observations (xh, xr) are not stored — only representations are saved per model.
    np.savez(
        run_dir / "sim_data.npz",
        z_s=data["z_s"],
        z_h=data["z_h"],
        z_r=data["z_r"],
        b=data["b"],
        cohort=data["cohort"].astype(str),
        w_s=data["weights"]["w_s"],
        w_h=data["weights"]["w_h"],
        w_r=data["weights"]["w_r"],
        w_b=data["weights"]["w_b"],
    )

    xh = {c: data["xh"][mask] for c, mask in cohort_masks.items()}
    xr = {c: data["xr"][mask] for c, mask in cohort_masks.items()}

    for mc in model_configs:
        print(f"[{run_id}] Training {mc.name}...", flush=True)
        model_dir = run_dir / mc.name
        model_dir.mkdir(exist_ok=True)

        if mc.model_type == "cca":
            _train_cca(mc, xh, xr, model_dir)
        elif mc.model_type == "jive":
            _train_jive(mc, xh, xr, model_dir)
        elif mc.model_type == "clip":
            _train_clip(mc, xh, xr, model_dir, seed=sim_config.seed)
        elif mc.model_type == "dssl":
            _train_dssl(mc, xh, xr, model_dir, seed=sim_config.seed)
        else:
            raise ValueError(f"Unsupported model_type: {mc.model_type!r}")

        with open(model_dir / "meta.json", "w") as f:
            json.dump(asdict(mc), f, indent=2)

        print(f"[{run_id}]   {mc.name} saved.", flush=True)


def _train_cca(
    mc: ModelConfig,
    xh: dict[str, np.ndarray],
    xr: dict[str, np.ndarray],
    model_dir: Path,
) -> None:
    model = CCA(**mc.hyperparams)
    model.fit(xh["A"], xr["A"])
    np.savez(
        model_dir / "latents.npz",
        **{f"Zh_{c}": model.encode_h(xh[c]) for c in COHORTS},
        **{f"Zr_{c}": model.encode_r(xr[c]) for c in COHORTS},
    )
    joblib.dump(model, model_dir / "model.joblib")


def _train_jive(
    mc: ModelConfig,
    xh: dict[str, np.ndarray],
    xr: dict[str, np.ndarray],
    model_dir: Path,
) -> None:
    model = JIVE(**mc.hyperparams)
    model.fit(xh["A"], xr["A"])
    np.savez(
        model_dir / "latents.npz",
        **{f"Zh_{c}": model.encode_h(xh[c]) for c in COHORTS},
        **{f"Zr_{c}": model.encode_r(xr[c]) for c in COHORTS},
    )
    gamma = compute_sharedness_scores(model.loadings)
    np.save(model_dir / "gamma.npy", gamma)
    joblib.dump(model, model_dir / "model.joblib")


def _train_clip(
    mc: ModelConfig,
    xh: dict[str, np.ndarray],
    xr: dict[str, np.ndarray],
    model_dir: Path,
    seed: int = 0,
) -> None:
    model = CLIP(**mc.hyperparams, seed=seed)
    model.fit(xh["A"], xr["A"])
    np.savez(
        model_dir / "latents.npz",
        **{f"Zh_{c}": model.encode_h(xh[c]) for c in COHORTS},
        **{f"Zr_{c}": model.encode_r(xr[c]) for c in COHORTS},
    )
    # CLIP is non-factorized: no gamma.npy
    torch.save({"enc_h": model._enc_h.state_dict(),
                "enc_r": model._enc_r.state_dict()}, model_dir / "model.pt")


def _train_dssl(
    mc: ModelConfig,
    xh: dict[str, np.ndarray],
    xr: dict[str, np.ndarray],
    model_dir: Path,
    seed: int = 0,
) -> None:
    model = DisentangledSSL(**mc.hyperparams, seed=seed)
    model.fit(xh["A"], xr["A"])
    np.savez(
        model_dir / "latents.npz",
        **{f"Zh_{c}": model.encode_h(xh[c]) for c in COHORTS},
        **{f"Zr_{c}": model.encode_r(xr[c]) for c in COHORTS},
    )
    # Hard binary sharedness: first latent_dim dims are shared (gamma=1),
    # last latent_dim dims are modality-specific (gamma=0).
    k = mc.hyperparams["latent_dim"]
    gamma = np.array([1.0] * k + [0.0] * k)
    np.save(model_dir / "gamma.npy", gamma)
    torch.save({"enc_h_c": model._enc_h_c.state_dict(),
                "enc_r_c": model._enc_r_c.state_dict(),
                "enc_h_s": model._enc_h_s.state_dict(),
                "enc_r_s": model._enc_r_s.state_dict()}, model_dir / "model.pt")


def generate_run_data(
    sim_config: SimulatorConfig,
    output_dir: str | Path,
    run_id: str,
) -> None:
    """Generate synthetic data for one run and save to disk.

    Saves latents, weights, metadata, and observations (xh, xr) so that
    individual model trainings can be dispatched independently via the
    GPU worker pool.
    """
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{run_id}] Generating synthetic data...", flush=True)
    data = simulate_run(sim_config)

    cohort_masks = {c: data["cohort"] == c for c in COHORTS}

    with open(run_dir / "metadata.json", "w") as f:
        json.dump({"run_id": run_id, "sim_config": asdict(sim_config)}, f, indent=2)

    np.savez(
        run_dir / "sim_data.npz",
        z_s=data["z_s"], z_h=data["z_h"], z_r=data["z_r"], b=data["b"],
        cohort=data["cohort"].astype(str),
        w_s=data["weights"]["w_s"], w_h=data["weights"]["w_h"],
        w_r=data["weights"]["w_r"], w_b=data["weights"]["w_b"],
    )

    np.savez(
        run_dir / "observations.npz",
        **{f"xh_{c}": data["xh"][mask] for c, mask in cohort_masks.items()},
        **{f"xr_{c}": data["xr"][mask] for c, mask in cohort_masks.items()},
    )


def train_single_model(
    run_dir: str | Path,
    mc: ModelConfig,
    seed: int,
) -> None:
    """Train one model on pre-generated observations saved to disk.

    Expects ``run_dir/observations.npz`` to exist (created by
    ``generate_run_data``).
    """
    run_dir = Path(run_dir)
    obs = np.load(run_dir / "observations.npz")
    xh = {c: obs[f"xh_{c}"] for c in COHORTS}
    xr = {c: obs[f"xr_{c}"] for c in COHORTS}

    model_dir = run_dir / mc.name
    model_dir.mkdir(exist_ok=True)

    if mc.model_type == "cca":
        _train_cca(mc, xh, xr, model_dir)
    elif mc.model_type == "jive":
        _train_jive(mc, xh, xr, model_dir)
    elif mc.model_type == "clip":
        _train_clip(mc, xh, xr, model_dir, seed=seed)
    elif mc.model_type == "dssl":
        _train_dssl(mc, xh, xr, model_dir, seed=seed)
    else:
        raise ValueError(f"Unsupported model_type: {mc.model_type!r}")

    with open(model_dir / "meta.json", "w") as f:
        json.dump(asdict(mc), f, indent=2)


def is_run_trained(run_dir: str | Path, model_configs: list[ModelConfig]) -> bool:
    """Return True if every model in ``model_configs`` finished training.

    ``metadata.json`` is written before any model is trained, so it does not
    mark a run as complete. Each model writes ``latents.npz`` and then
    ``meta.json`` once training finishes.

    Parameters
    ----------
    run_dir : str or Path
        Run directory containing one subdirectory per model.
    model_configs : list[ModelConfig]
        Models expected for this run.

    Returns
    -------
    bool
        True if ``latents.npz`` and ``meta.json`` exist for every model.
    """
    run_dir = Path(run_dir)
    return all(
        (run_dir / mc.name / "latents.npz").exists()
        and (run_dir / mc.name / "meta.json").exists()
        for mc in model_configs
    )


def _gpu_worker_fn(gpu_id: int, job_queue: mp.Queue) -> None:
    """Worker process bound to one GPU via CUDA_VISIBLE_DEVICES.

    Pulls (run_dir, ModelConfig, seed) jobs from a shared queue until
    it receives a ``None`` sentinel.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        job = job_queue.get()
        if job is None:
            break
        run_dir_str, mc, seed = job
        run_dir = Path(run_dir_str)
        print(f"[GPU {gpu_id}] {run_dir.name}/{mc.name}", flush=True)
        train_single_model(run_dir, mc, seed)

        # Release GPU memory between trainings.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def train_runs_gpu_pool(
    run_configs: list[tuple[SimulatorConfig, str]],
    model_configs: list[ModelConfig],
    output_dir: str | Path,
    n_gpus: int = 8,
) -> None:
    """Train all models across multiple runs using a GPU worker pool.

    Phase 1: generate synthetic data for every run (CPU, sequential).
    Phase 2: enqueue all (run, model) training jobs and distribute them
             across ``n_gpus`` worker processes, one per GPU.

    Parameters
    ----------
    run_configs : list of (SimulatorConfig, run_id) tuples
        Each entry defines one synthetic run to generate and train on.
    model_configs : list[ModelConfig]
        Models to train per run.
    output_dir : str or Path
        Root directory for representations output.
    n_gpus : int
        Number of GPU workers to spawn. Each is bound to
        ``CUDA_VISIBLE_DEVICES=<gpu_id>``.
    """
    output_dir = Path(output_dir)

    # Phase 1: generate all synthetic data (CPU only).
    print(f"Generating data for {len(run_configs)} runs...", flush=True)
    for sim_cfg, run_id in run_configs:
        generate_run_data(sim_cfg, output_dir, run_id)

    # Phase 2: build job queue and spawn GPU workers.
    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    n_jobs = 0
    for sim_cfg, run_id in run_configs:
        run_dir = output_dir / run_id
        for mc in model_configs:
            job_queue.put((str(run_dir), mc, sim_cfg.seed))
            n_jobs += 1

    # Sentinel values to signal workers to stop.
    for _ in range(n_gpus):
        job_queue.put(None)

    print(f"Starting {n_gpus} GPU workers for {n_jobs} training jobs...",
          flush=True)
    workers = []
    for gpu_id in range(n_gpus):
        p = ctx.Process(target=_gpu_worker_fn, args=(gpu_id, job_queue))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    # Clean up observation files (large, no longer needed).
    for _, run_id in run_configs:
        obs_path = output_dir / run_id / "observations.npz"
        if obs_path.exists():
            obs_path.unlink()

    print(f"GPU pool complete: {n_jobs} models trained.", flush=True)


def main() -> None:
    """CLI entrypoint: train all models for one synthetic run."""
    parser = argparse.ArgumentParser(description="DECAT representation training")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--output-dir", default=f"{DEFAULT_RESULTS}/representations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--d-h", type=int, default=1024)
    parser.add_argument("--d-r", type=int, default=1024)
    parser.add_argument("--n-a", type=int, default=2500)
    parser.add_argument("--n-eval", type=int, default=500,
                        help="Cohort size for A_prime, B, C")
    parser.add_argument("--beta-b", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Feature-level proxy entanglement (affects representation learning)")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="Proxy misalignment parameter")
    parser.add_argument("--batch-align", type=float, default=0.0,
                        help="Alignment of mu_b with w_b (direct confounding detection)")
    parser.add_argument("--proxy-shift-align", type=float, default=0.0,
                        help="Alignment of mu_b with w_h or w_r (proxy detection)")
    parser.add_argument("--proxy-shift-target", type=str, default="h",
                        choices=["h", "r"],
                        help="Modality target for proxy_shift_align")
    args = parser.parse_args()

    cfg = SimulatorConfig(
        latent_dim=args.latent_dim,
        d_h=args.d_h,
        d_r=args.d_r,
        n_a=args.n_a,
        n_a_prime=args.n_eval,
        n_b=args.n_eval,
        n_c=args.n_eval,
        beta_b=args.beta_b,
        alpha_s=0.0, alpha_h=0.0, alpha_r=0.0, alpha_b=0.0,
        gamma_h=args.gamma, gamma_r=args.gamma,
        eta=args.eta,
        batch_align=args.batch_align,
        proxy_shift_align=args.proxy_shift_align,
        proxy_shift_target=args.proxy_shift_target,
        seed=args.seed,
    )
    train_run(cfg, default_model_configs(args.latent_dim), args.output_dir, args.run_id)


if __name__ == "__main__":
    main()
