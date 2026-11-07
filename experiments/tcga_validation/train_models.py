"""TCGA Validation — Model Training.

Trains all 7 representation models on Cohort A (seed-0, 55% of paired patients):
  CCA, JIVE        — linear, CPU (synchronous, seconds)
  CLIP             — nonlinear entangled, GPU
  DSSL ×4          — nonlinear factorized, GPU (β=λ ∈ {0.01, 0.1, 1.0, 10.0})

GPU models run in parallel via a worker pool (one process per GPU), matching
the pattern in pipeline/train_models.py.  CPU models run first synchronously.

Outputs per model::

    <OUT_DIR>/representations/<name>/full_latents.npz   — (N, dim) for all patients
    <OUT_DIR>/representations/<name>/gamma.npy           — factorized models only
    <OUT_DIR>/representations/<name>/meta.json

Usage::

    python train_models.py
"""

import json
import os
import sys
import time
from pathlib import Path
import multiprocessing as mp

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

OUT_DIR    = os.environ.get('DECAT_OUT_DIR', './data/tcga')  # override with --out-dir
LATENT_DIM = 50
N_GPUS     = 4

# Epoch counts from synthetic experiment calibration
GPU_MODEL_CONFIGS = [
    {
        "name":       "clip_K50",
        "model_type": "clip",
        "kwargs":     {"latent_dim": LATENT_DIM, "n_epochs": 200, "batch_size": 256},
    },
    {
        "name":       "dssl_l0.01_b0.01_K50",
        "model_type": "dssl",
        "kwargs":     {"latent_dim": LATENT_DIM, "beta": 0.01, "lambda_orth": 0.01,
                       "n_epochs_shared": 125, "n_epochs_specific": 50, "batch_size": 256},
    },
    {
        "name":       "dssl_l0.1_b0.1_K50",
        "model_type": "dssl",
        "kwargs":     {"latent_dim": LATENT_DIM, "beta": 0.1, "lambda_orth": 0.1,
                       "n_epochs_shared": 125, "n_epochs_specific": 50, "batch_size": 256},
    },
    {
        "name":       "dssl_l1.0_b1.0_K50",
        "model_type": "dssl",
        "kwargs":     {"latent_dim": LATENT_DIM, "beta": 1.0, "lambda_orth": 1.0,
                       "n_epochs_shared": 75, "n_epochs_specific": 50, "batch_size": 256},
    },
    {
        "name":       "dssl_l10.0_b10.0_K50",
        "model_type": "dssl",
        "kwargs":     {"latent_dim": LATENT_DIM, "beta": 10.0, "lambda_orth": 10.0,
                       "n_epochs_shared": 75, "n_epochs_specific": 50, "batch_size": 256},
    },
]


def ts(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _encode_and_save(model, name: str, model_type: str, kwargs: dict,
                     he_full: np.ndarray, rna_full: np.ndarray,
                     model_dir: Path) -> None:
    """Encode all patients, save full_latents.npz, gamma, and meta."""
    Zh = model.encode_h(he_full)
    Zr = model.encode_r(rna_full)
    np.savez(model_dir / "full_latents.npz", Zh=Zh, Zr=Zr)
    ts(f"  [{name}] encoded {len(Zh)} patients → full_latents.npz ({Zh.shape[1]}-dim)")

    if model_type == "jive":
        from metrics.delta_shared import compute_sharedness_scores
        gamma = compute_sharedness_scores(model.loadings)
        np.save(model_dir / "gamma.npy", gamma)
        ts(f"  [{name}] gamma saved ({gamma.shape})")
    elif model_type == "dssl":
        k     = kwargs["latent_dim"]
        gamma = np.array([1.0] * k + [0.0] * k, dtype=np.float32)
        np.save(model_dir / "gamma.npy", gamma)
        ts(f"  [{name}] gamma saved ({gamma.shape})")

    with open(model_dir / "meta.json", "w") as f:
        json.dump({"name": name, "model_type": model_type, **kwargs}, f, indent=2)


def _train_cpu(name: str, model_type: str, kwargs: dict,
               he_a: np.ndarray, rna_a: np.ndarray,
               he_full: np.ndarray, rna_full: np.ndarray,
               model_dir: Path) -> None:
    """Train one CPU model (CCA or JIVE) synchronously."""
    from models.cca import CCA
    from models.jive import JIVE

    model_dir.mkdir(parents=True, exist_ok=True)
    if (model_dir / "full_latents.npz").exists():
        ts(f"  [{name}] already done, skipping")
        return

    t0    = time.time()
    model = CCA(**kwargs) if model_type == "cca" else JIVE(**kwargs)
    model.fit(he_a, rna_a)
    ts(f"  [{name}] fit in {time.time()-t0:.1f}s")
    _encode_and_save(model, name, model_type, kwargs, he_full, rna_full, model_dir)
    ts(f"  [{name}] complete ({time.time()-t0:.1f}s total)")


def _gpu_worker_fn(gpu_id: int, job_queue: mp.Queue) -> None:
    """Worker bound to one GPU — drains the shared job queue.

    Mirrors pipeline/train_models.py:_gpu_worker_fn.
    """
    sys.path.insert(0, str(Path(__file__).parents[2]))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        job = job_queue.get()
        if job is None:
            break
        cfg, he_a, rna_a, he_full, rna_full, model_dir_str = job
        name       = cfg["name"]
        model_type = cfg["model_type"]
        kwargs     = {**cfg["kwargs"], "device": "cuda:0"}  # remapped inside process
        model_dir  = Path(model_dir_str)

        if (model_dir / "full_latents.npz").exists():
            ts(f"  [GPU {gpu_id}] [{name}] already done, skipping")
            continue

        from models.clip import CLIP
        from models.disentangled_ssl import DisentangledSSL

        model_dir.mkdir(parents=True, exist_ok=True)
        t0    = time.time()
        ts(f"  [GPU {gpu_id}] [{name}] starting")
        model = CLIP(**kwargs) if model_type == "clip" else DisentangledSSL(**kwargs)
        model.fit(he_a, rna_a)
        ts(f"  [GPU {gpu_id}] [{name}] fit in {time.time()-t0:.1f}s")
        _encode_and_save(model, name, model_type, cfg["kwargs"], he_full, rna_full, model_dir)
        ts(f"  [GPU {gpu_id}] [{name}] done ({time.time()-t0:.1f}s total)")


def main() -> None:
    ts("=== TCGA Validation — Model Training ===")

    out_dir  = Path(OUT_DIR)
    repr_dir = out_dir / "representations"
    repr_dir.mkdir(parents=True, exist_ok=True)

    ts("Loading embeddings ...")
    he_full  = np.load(out_dir / "he_embeddings.npy")
    rna_full = np.load(out_dir / "rna_embeddings.npy")
    ts(f"  H&E: {he_full.shape},  RNA: {rna_full.shape}")

    ts("Loading Cohort A (seed=0) ...")
    with open(out_dir / "pooled_random_splits.json") as f:
        splits = json.load(f)
    s0    = splits[0]
    assert s0["seed"] == 0
    a_idx = np.array(s0["cohort_a"])
    he_a  = he_full[a_idx]
    rna_a = rna_full[a_idx]
    ts(f"  Cohort A: {len(a_idx)} patients")

    # CPU models: CCA and JIVE
    ts("Training CPU models (CCA, JIVE) ...")
    for model_type, name, kwargs in [
        ("cca",  f"cca_K{LATENT_DIM}",  {"n_components": LATENT_DIM}),
        ("jive", f"jive_K{LATENT_DIM}", {"n_components": LATENT_DIM}),
    ]:
        _train_cpu(name, model_type, kwargs, he_a, rna_a, he_full, rna_full,
                   repr_dir / name)

    # GPU models: CLIP and DSSL via worker pool
    # Pattern from pipeline/train_models.py: mp.get_context("spawn") avoids
    # setting a global start method while giving each worker an isolated CUDA context.
    n_gpus = min(N_GPUS, len(GPU_MODEL_CONFIGS))
    ts(f"Training GPU models ({len(GPU_MODEL_CONFIGS)} models, {n_gpus} GPUs) ...")

    ctx       = mp.get_context("spawn")
    job_queue = ctx.Queue()

    for cfg in GPU_MODEL_CONFIGS:
        job_queue.put((cfg, he_a, rna_a, he_full, rna_full,
                       str(repr_dir / cfg["name"])))
    for _ in range(n_gpus):
        job_queue.put(None)  # sentinels

    workers = []
    for gpu_id in range(n_gpus):
        p = ctx.Process(target=_gpu_worker_fn, args=(gpu_id, job_queue))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()
        if p.exitcode != 0:
            ts(f"  WARNING: worker exited with code {p.exitcode}")

    ts("=== Training complete ===")
    ts(f"  All models: {sorted(d.name for d in repr_dir.iterdir() if d.is_dir())}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--he-csv",  default=None)  # unused by train, accepted for pipeline compat
    parser.add_argument("--out-dir", default=None)
    a = parser.parse_args()
    if a.out_dir: OUT_DIR = a.out_dir
    main()
