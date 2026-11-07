"""Scenario evaluation pipeline for DECAT experiments.

Loads saved representations, generates outcomes under predefined alpha
configurations, runs the full DECAT metric pipeline (Stages I-IV), classifies
scenarios, and saves results.

Results are written as a JSON array of flat dicts, one record per
(model, alpha_config, n_eval, modality) combination, suitable for direct
loading into pandas via ``pd.read_json``.

Loop structure
--------------
Each (model, n_eval, alpha_config) combination is evaluated independently
via ``_evaluate_combination()``.  When ``n_jobs > 1``, these combinations
run in parallel (threading backend, shared memory) with deterministic
per-combination seeds derived via SHA-256 hashing.  Inner permutation /
bootstrap loops run sequentially within each combination to avoid
oversubscription.
"""

import argparse
import hashlib
import json
import os
import time
import numpy as np
from dataclasses import dataclass, field, replace
from pathlib import Path

from joblib import Parallel, delayed

DEFAULT_RESULTS = os.environ.get("DECAT_RESULTS_DIR", "./results")

from sklearn.model_selection import StratifiedShuffleSplit

from simulator.outcome_model import generate_outcomes
from metrics.anorm import compute_anorm_null
from metrics.bnorm import compute_bnorm_null
from metrics.delta_shared import compute_delta_shared_bootstrap, soft_mask_representations
from metrics.transfer import compute_transfer_ci, compute_transfer_null
from metrics.quantile import compute_dquantile_task
from evaluation.probes import cv_auroc, make_cv_splits
from evaluation.prototypes import compute_prototypes
from decision.scenario_classifier import MetricBundle, classify_scenario, is_shared_dominant, is_modality_dominant

COHORTS_EVAL = ("A_prime", "B", "C")


@dataclass
class AlphaConfig:
    """Outcome configuration for one evaluation task.

    Attributes
    ----------
    config_id : str
        Short identifier (e.g. ``"s1_as0.5"``).
    alpha_s : float
        Shared-biology outcome coefficient.
    alpha_h : float
        H&E-specific outcome coefficient.
    alpha_r : float
        RNA-specific outcome coefficient.
    alpha_b : float
        Outcome-level confounding coefficient.
    gamma : float
        Feature-level proxy severity. The representation is already fixed
        at evaluation time; this field is recorded for bookkeeping only.
    eta : float
        Proxy misalignment parameter (bookkeeping only).
    sigma_y : float
        Outcome noise standard deviation.
    """

    config_id: str
    alpha_s: float = 0.0
    alpha_h: float = 0.0
    alpha_r: float = 0.0
    alpha_b: float = 0.0
    gamma: float = 0.0
    eta: float = 0.0
    sigma_y: float = 0.5


@dataclass
class EvalConfig:
    """Evaluation hyperparameters.

    Attributes
    ----------
    n_eval_sizes : list[int]
        Evaluation cohort sizes swept per run.
    n_bootstrap : int
        Bootstrap replicates for P_transfer and Delta_shared CIs.
    n_permutations : int
        Permutation replicates for null distributions.
    n_directions : int
        Random-projection directions for D_quantile (random-projection variant).
    n_proto_clusters : int
        Number of k-means clusters per seed for D_quantile prototype variant.
    n_proto_seeds : int
        Independent k-means seeds for D_quantile prototype variant.
    n_folds : int
        CV folds for AUROC and Delta_shared computation.
    n_jobs : int
        Parallel workers for permutation loops (1 = sequential).
    """

    n_eval_sizes: list[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    n_bootstrap: int = 1000
    n_permutations: int = 1000
    n_directions: int = 20
    n_proto_clusters: int = 5
    n_proto_seeds: int = 5
    n_folds: int = 5
    n_jobs: int = 1
    dq_probe_backend: str = 'logistic'


def make_alpha_configs() -> list[AlphaConfig]:
    """Construct the baseline-compatible alpha configurations.

    These configs are applicable to baseline runs (gamma_h=gamma_r=0) and
    do not require proxy-entangled representations.  Proxy-specific configs
    (S2A, Transition B, Mixed C, proxy calibration) are returned by
    ``make_proxy_alpha_configs()``.

    Breakdown:
    - S1 shared biology:           4  (alpha_s ∈ {0.25,0.5,0.8,1.1})
    - S3 noise null:                2
    - S4h H&E-specific:             4  (alpha_h ∈ {0.25,0.5,0.8,1.1})
    - S4r RNA-specific:             4  (alpha_r ∈ {0.25,0.5,0.8,1.1})
    - S2B outcome confounding:      4  (alpha_b ∈ {0.25,0.5,0.8,1.1})
    - Mixed type A (shared+mod-spec): 3  (2 asymmetric + 1 balanced)
    - Mixed type B (two mod-spec):  2
    - Mixed type D (shared+conf):   2
    - Transition A (bio ambiguity): 3

    Returns
    -------
    list[AlphaConfig]
    """
    configs: list[AlphaConfig] = []

    for alpha_s in (0.25, 0.5, 0.8, 1.1):
        configs.append(AlphaConfig(f"s1_as{alpha_s}", alpha_s=alpha_s))

    configs.append(AlphaConfig("s3_noise_0"))
    configs.append(AlphaConfig("s3_noise_1"))

    for alpha_h in (0.25, 0.5, 0.8, 1.1):
        configs.append(AlphaConfig(f"s4h_ah{alpha_h}", alpha_h=alpha_h))

    for alpha_r in (0.25, 0.5, 0.8, 1.1):
        configs.append(AlphaConfig(f"s4r_ar{alpha_r}", alpha_r=alpha_r))

    for alpha_b in (0.25, 0.5, 0.8, 1.1):
        configs.append(AlphaConfig(f"s2b_ab{alpha_b}", alpha_b=alpha_b))

    # Mixed type A — shared + modality-specific.
    # Ground truth: RNA is S1, H&E is S4. Tests factorized model resolution.
    # The two asymmetric configs (alpha_s >> alpha_h) have shared signal dominating
    # the localization test, so H&E routes to S1 rather than S4 — joint resolution=0%.
    # The balanced config (alpha_s=alpha_h=0.8) gives delta_shared a genuine chance
    # to route H&E as modality-dominant while RNA stays S1, testing whether DECAT
    # can simultaneously report S1-for-RNA and S4-for-H&E.
    configs.append(AlphaConfig("mixed_a_as1.1_ah0.3", alpha_s=1.1, alpha_h=0.3))
    configs.append(AlphaConfig("mixed_a_as0.8_ah0.5", alpha_s=0.8, alpha_h=0.5))
    configs.append(AlphaConfig("mixed_a_as0.8_ah0.8", alpha_s=0.8, alpha_h=0.8))

    # Mixed type B — two modality-specific.
    # Ground truth: S4 for both modalities, asymmetric signal strength.
    configs.append(AlphaConfig("mixed_b_ah1.1_ar0.3", alpha_h=1.1, alpha_r=0.3))
    configs.append(AlphaConfig("mixed_b_ah0.8_ar0.5", alpha_h=0.8, alpha_r=0.5))

    # Mixed type D — shared biology + outcome-level confounding.
    configs.append(AlphaConfig("mixed_d_as1.1_ab0.3", alpha_s=1.1, alpha_b=0.3))
    configs.append(AlphaConfig("mixed_d_as0.8_ab0.5", alpha_s=0.8, alpha_b=0.5))

    # Transition A — biological ambiguity.
    # alpha_s ≈ alpha_h ≈ alpha_r, no proxy, no confounding.  Magnitudes from
    # {0.4, 0.6} with one component dominant per config; tests whether DECAT
    # correctly returns indeterminate when no mechanism clearly dominates.
    configs.append(AlphaConfig("trans_a_sh_dom",  alpha_s=0.6, alpha_h=0.4, alpha_r=0.4))
    configs.append(AlphaConfig("trans_a_h_dom",   alpha_s=0.4, alpha_h=0.6, alpha_r=0.4))
    configs.append(AlphaConfig("trans_a_r_dom",   alpha_s=0.4, alpha_h=0.4, alpha_r=0.6))

    return configs


def make_proxy_alpha_configs(gamma: float, eta: float) -> list[AlphaConfig]:
    """Construct the 17 proxy-specific alpha configs for one proxy run group.

    These configs require proxy-entangled representations (gamma_h > 0 or
    gamma_r > 0 in the run's SimulatorConfig).  For proxy runs the evaluation
    uses these 17 configs plus the 4 S1 configs from make_alpha_configs()
    (to assess false-shared-claim rate on proxy-contaminated data), totalling
    21 configs per proxy run (spec §8: 17 + 4 = 21).

    Parameters
    ----------
    gamma : float
        Proxy strength matching the run group's gamma_h (= gamma_r for
        dual-modality groups).
    eta : float
        Proxy misalignment matching the run group's eta.

    Breakdown (17 total):
    - S2A proxy misaligned:          4  (alpha_h × 4 strengths, gamma/eta from group)
    - Transition B (conf. ambiguity): 3  (bio ambiguity + proxy + confounding)
    - Mixed type C (shared + proxy):  2
    - S2C aligned proxy (η=0):        2  (gamma_h ∈ {0.5,1.0}, eta=0)
    - S2D shared proxy (dual-modal):  2  (gamma_h=gamma_r ∈ {0.5,1.0}, eta=0)
    - Proxy calibration sweep:        4  (alpha_h=0.8, varying gamma/eta)

    Returns
    -------
    list[AlphaConfig]
    """
    configs: list[AlphaConfig] = []

    # S2A — feature-level proxy misalignment.
    # Outcome depends on modality-specific signal in the presence of proxy
    # contamination; gamma/eta match the run group's representation regime.
    for alpha_h in (0.25, 0.5, 0.8, 1.1):
        configs.append(AlphaConfig(
            f"s2a_ah{alpha_h}_g{gamma}_e{eta}",
            alpha_h=alpha_h, gamma=gamma, eta=eta,
        ))

    # Transition B — confounded ambiguity.
    # Biological ambiguity (alpha_s ≈ alpha_h ≈ alpha_r) layered with
    # proxy (gamma=0.3, eta=0.3) and confounding (alpha_b=0.3).
    for as_, ah, ar in ((0.6, 0.4, 0.4), (0.4, 0.6, 0.4), (0.4, 0.4, 0.6)):
        configs.append(AlphaConfig(
            f"trans_b_as{as_}_ah{ah}_ar{ar}",
            alpha_s=as_, alpha_h=ah, alpha_r=ar,
            alpha_b=0.3, gamma=0.3, eta=0.3,
        ))

    # Mixed type C — shared biology + proxy.
    # alpha_s dominant, alpha_h secondary with proxy contamination; eta=0.3
    # creates partial misalignment so proxy instability is partially detectable.
    configs.append(AlphaConfig(
        "mixed_c_as1.1_ah0.3_g0.5_e0.3",
        alpha_s=1.1, alpha_h=0.3, gamma=0.5, eta=0.3,
    ))
    configs.append(AlphaConfig(
        "mixed_c_as0.8_ah0.5_g1.0_e0.3",
        alpha_s=0.8, alpha_h=0.5, gamma=1.0, eta=0.3,
    ))

    # S2C — aligned proxy, η=0.
    # Proxy perfectly aligned with batch axis; non-identifiable via A_norm/B_norm.
    # Two gamma levels test detection difficulty.
    configs.append(AlphaConfig("s2c_ah0.8_g0.5_e0", alpha_h=0.8, gamma=0.5, eta=0.0))
    configs.append(AlphaConfig("s2c_ah0.8_g1.0_e0", alpha_h=0.8, gamma=1.0, eta=0.0))

    # S2D — shared proxy, dual-modality entanglement.
    # alpha_h=alpha_r=0.8, gamma applies to both modalities.
    configs.append(AlphaConfig("s2d_ah0.8_ar0.8_g0.5_e0",
                               alpha_h=0.8, alpha_r=0.8, gamma=0.5, eta=0.0))
    configs.append(AlphaConfig("s2d_ah0.8_ar0.8_g1.0_e0",
                               alpha_h=0.8, alpha_r=0.8, gamma=1.0, eta=0.0))

    # Proxy calibration sweep.
    # alpha_h=0.8 fixed; varies gamma/eta to sweep proxy stability boundary.
    # Config 1 (gamma=0) is a S4 baseline on proxy representations;
    # configs 2-4 test detection across alignment conditions.
    configs.append(AlphaConfig("proxy_cal_ah0.8_g0_e0",   alpha_h=0.8, gamma=0.0, eta=0.0))
    configs.append(AlphaConfig("proxy_cal_ah0.8_g0.5_e0", alpha_h=0.8, gamma=0.5, eta=0.0))
    configs.append(AlphaConfig("proxy_cal_ah0.8_g0.5_e0.3", alpha_h=0.8, gamma=0.5, eta=0.3))
    configs.append(AlphaConfig("proxy_cal_ah0.8_g0.5_e0.6", alpha_h=0.8, gamma=0.5, eta=0.6))

    return configs


def _combo_seed(run_seed: int, model_name: str, alpha_id: str, n_eval: int) -> int:
    """Derive a deterministic seed for one (model, alpha, n_eval) combination.

    Uses SHA-256 hashing so seeds are independent of execution order,
    enabling safe parallelism across combinations.
    """
    key = f"{run_seed}:{model_name}:{alpha_id}:{n_eval}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (1 << 63)


def _evaluate_combination(
    run_id: str,
    model_name: str,
    is_factorized: bool,
    Zh: dict[str, np.ndarray],
    Zr: dict[str, np.ndarray],
    gamma: np.ndarray | None,
    cohort_latents: dict,
    weights: dict,
    cohort_sizes: dict[str, int],
    n_eval: int,
    alpha_cfg: AlphaConfig,
    eval_config: EvalConfig,
    combo_seed: int,
) -> list[dict]:
    """Evaluate one (model, n_eval, alpha_cfg) combination for both modalities.

    This is the parallelisable unit: each call is independent and uses a
    deterministic seed derived from the combination key, not from a shared
    RNG chain.

    Returns
    -------
    list of dict
        Two records (one per modality).
    """
    rng_alpha = np.random.default_rng(combo_seed)

    y_full = {
        c: generate_outcomes(
            cohort_latents[c], weights,
            alpha_s=alpha_cfg.alpha_s,
            alpha_h=alpha_cfg.alpha_h,
            alpha_r=alpha_cfg.alpha_r,
            alpha_b=alpha_cfg.alpha_b,
            gamma_h=alpha_cfg.gamma, gamma_r=alpha_cfg.gamma,
            eta=alpha_cfg.eta,
            sigma_y=alpha_cfg.sigma_y,
            rng=np.random.default_rng(rng_alpha.integers(1 << 63)),
        )[0]
        for c in COHORTS_EVAL
    }

    idx = {
        c: _stratified_subsample(
            cohort_sizes[c], y_full[c], n_eval,
            rng=np.random.default_rng(rng_alpha.integers(1 << 63)),
        )
        for c in COHORTS_EVAL
    }
    y = {c: y_full[c][idx[c]] for c in COHORTS_EVAL}

    stage1 = _compute_stage1(
        _std(Zh["A_prime"][idx["A_prime"]]),
        _std(Zr["A_prime"][idx["A_prime"]]),
        eval_config,
        rng=np.random.default_rng(rng_alpha.integers(1 << 63)),
    )

    records = []
    for modality in ("h", "r"):
        rng_mod = np.random.default_rng(rng_alpha.integers(1 << 63))
        Z_raw = Zh if modality == "h" else Zr

        Z_eval_raw = {c: Z_raw[c][idx[c]] for c in COHORTS_EVAL}
        Z = {c: _std(Z_eval_raw[c]) for c in COHORTS_EVAL}

        if is_factorized:
            Z_shared   = {c: soft_mask_representations(Z[c], gamma)[0]
                          for c in COHORTS_EVAL}
            Z_specific = {c: soft_mask_representations(Z[c], gamma)[1]
                          for c in COHORTS_EVAL}
            Z_shared_raw   = {c: soft_mask_representations(Z_eval_raw[c], gamma)[0]
                              for c in COHORTS_EVAL}
            Z_specific_raw = {c: soft_mask_representations(Z_eval_raw[c], gamma)[1]
                              for c in COHORTS_EVAL}
            ds = compute_delta_shared_bootstrap(
                Z_shared["A_prime"], Z_specific["A_prime"], y["A_prime"],
                n_folds=eval_config.n_folds,
                n_bootstrap=eval_config.n_bootstrap,
                n_permutations=eval_config.n_permutations,
                n_jobs=eval_config.n_jobs,
                rng=np.random.default_rng(rng_mod.integers(1 << 63)),
                standardize=False,
            )
            signal_present = ds["signal_present"]
            z_sig, loc = _select_signal_component(ds, Z_shared, Z_specific)
            z_sig_raw, _ = _select_signal_component(
                ds, Z_shared_raw, Z_specific_raw
            )
        else:
            ds = None
            loc = None
            signal_present, _ = _signal_gate_entangled(
                Z["A_prime"], y["A_prime"], eval_config,
                rng=np.random.default_rng(rng_mod.integers(1 << 63)),
            )
            z_sig = Z
            z_sig_raw = Z_eval_raw

        s4 = _compute_stage4(
            z_sig["A_prime"],     y["A_prime"],
            z_sig["B"],           y["B"],
            z_sig["C"],
            z_sig_raw["A_prime"],
            z_sig_raw["B"],
            z_sig_raw["C"],
            eval_config,
            signal_present=signal_present,
            rng=np.random.default_rng(rng_mod.integers(1 << 63)),
        )

        bundle = MetricBundle(
            anorm=stage1["anorm"],
            anorm_null_upper=stage1["anorm_null_upper"],
            signal_present=signal_present,
            transfer_ci_lower=s4["transfer_ci_lower"],
            transfer_null_upper=s4["transfer_null_upper"],
            dquantile_task=s4["dquantile_task"],
            dquantile_task_null_upper=s4["dquantile_task_null_upper"],
            is_factorized=is_factorized,
            delta_ci_lower=ds["ci_lower"] if ds else None,
            delta_ci_upper=ds["ci_upper"] if ds else None,
            delta_null_lower=ds.get("delta_null_lower") if ds else None,
            delta_null_upper=ds.get("delta_null_upper") if ds else None,
        )
        scenario = classify_scenario(bundle)

        records.append(_build_record(
            run_id, model_name, alpha_cfg, n_eval, modality,
            stage1, ds, signal_present, loc, s4, scenario,
        ))

    return records


def evaluate_run(
    run_id: str,
    repr_dir: str | Path,
    alpha_configs: list[AlphaConfig],
    eval_config: EvalConfig,
    output_dir: str | Path,
    seed: int = 0,
) -> list[dict]:
    """Run the full DECAT evaluation pipeline for one synthetic run.

    When ``eval_config.n_jobs > 1``, the (model x n_eval x alpha_config)
    combinations are evaluated in parallel using a threading backend.
    Each combination receives a deterministic seed derived from
    ``(seed, model_name, alpha_id, n_eval)`` via SHA-256, so results are
    reproducible regardless of execution order.

    Parameters
    ----------
    run_id : str
        Identifier matching a subdirectory in repr_dir.
    repr_dir : str or Path
        Root directory of saved representations (contains run_id subdir).
    alpha_configs : list[AlphaConfig]
        Outcome configurations to evaluate.
    eval_config : EvalConfig
        Evaluation hyperparameters.
    output_dir : str or Path
        Directory to write the results JSON.
    seed : int
        Base seed for all RNG streams in this run.

    Returns
    -------
    list of dict
        One record per (model, alpha_config, n_eval, modality).
    """
    run_dir = Path(repr_dir) / run_id

    sim = np.load(run_dir / "sim_data.npz", allow_pickle=True)
    cohort_labels = sim["cohort"]
    cohort_masks = {c: cohort_labels == c for c in COHORTS_EVAL}
    cohort_latents = {
        c: {
            "z_s": sim["z_s"][m], "z_h": sim["z_h"][m],
            "z_r": sim["z_r"][m], "b":   sim["b"][m],
        }
        for c, m in cohort_masks.items()
    }
    weights = {k: sim[k] for k in ("w_s", "w_h", "w_r", "w_b")}
    cohort_sizes = {c: int(np.sum(m)) for c, m in cohort_masks.items()}

    model_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )

    # Load all model data upfront so parallel workers can share it.
    print(f"[{run_id}] Loading {len(model_dirs)} models...", flush=True)
    models: list[tuple[str, bool, dict, dict, np.ndarray | None]] = []
    for model_dir in model_dirs:
        with open(model_dir / "meta.json") as f:
            meta = json.load(f)
        lats = np.load(model_dir / "latents.npz")
        Zh = {c: lats[f"Zh_{c}"] for c in COHORTS_EVAL}
        Zr = {c: lats[f"Zr_{c}"] for c in COHORTS_EVAL}
        gamma = np.load(model_dir / "gamma.npy") if meta["is_factorized"] else None
        models.append((meta["name"], meta["is_factorized"], Zh, Zr, gamma))

    # When using outer parallelism across combinations, disable inner
    # parallelism (permutation/bootstrap loops) to avoid oversubscription.
    inner_cfg = (
        replace(eval_config, n_jobs=1)
        if eval_config.n_jobs > 1
        else eval_config
    )

    combos = [
        (model_name, is_fac, Zh, Zr, gam, n_eval, alpha_cfg)
        for model_name, is_fac, Zh, Zr, gam in models
        for n_eval in eval_config.n_eval_sizes
        for alpha_cfg in alpha_configs
    ]

    n_combos = len(combos)

    print(f"[{run_id}] Evaluating {n_combos} (model x n_eval x alpha) "
          f"combos, n_jobs={eval_config.n_jobs}...", flush=True)

    # Use n_jobs=1 here — outer combo-level threading adds GIL contention
    # overhead that outweighs any benefit when running inside loky workers.
    # Parallelism is provided by n_concurrent loky processes in
    # evaluate_runs_parallel instead.
    results = Parallel(
        n_jobs=1,
        verbose=10,
    )(
        delayed(_evaluate_combination)(
            run_id, model_name, is_fac, Zh, Zr, gam,
            cohort_latents, weights, cohort_sizes,
            n_eval, alpha_cfg, inner_cfg,
            _combo_seed(seed, model_name, alpha_cfg.config_id, n_eval),
        )
        for model_name, is_fac, Zh, Zr, gam, n_eval, alpha_cfg in combos
    )

    records = [r for batch in results for r in batch]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump([_serialize(r) for r in records], f)

    print(f"[{run_id}] {len(records)} records saved to {out_path}", flush=True)
    return records


def _std(Z: np.ndarray) -> np.ndarray:
    return (Z - Z.mean(0)) / Z.std(0).clip(min=1e-8)


def _stratified_subsample(
    n: int,
    y: np.ndarray,
    n_eval: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return stratified indices subsampled to n_eval.

    Parameters
    ----------
    n : int
        Total number of samples in the cohort.
    y : ndarray of shape (n,)
        Binary outcome labels used for stratification.
    n_eval : int
        Target subsample size. If n_eval >= n, all indices are returned.
    rng : np.random.Generator

    Returns
    -------
    ndarray of shape (min(n_eval, n),)
        Selected indices.
    """
    if n_eval >= n:
        return np.arange(n)
    try:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=n_eval,
            random_state=int(rng.integers(1 << 31)),
        )
        idx, _ = next(sss.split(np.zeros(n), y))
        return idx
    except ValueError:
        # Fallback for degenerate class distributions
        return rng.choice(n, size=n_eval, replace=False)


def _compute_stage1(
    zh_ap: np.ndarray,
    zr_ap: np.ndarray,
    cfg: EvalConfig,
    rng: np.random.Generator,
) -> dict:
    anorm = compute_anorm_null(
        zh_ap, zr_ap,
        n_permutations=cfg.n_permutations,
        rng=np.random.default_rng(rng.integers(1 << 63)),
    )
    protos = compute_prototypes(
        zh_ap, k=cfg.n_proto_clusters,
        rng=np.random.default_rng(rng.integers(1 << 63)),
    )
    bnorm = compute_bnorm_null(
        zh_ap, zr_ap, protos,
        n_permutations=cfg.n_permutations,
        rng=np.random.default_rng(rng.integers(1 << 63)),
    )
    return {
        "anorm":            float(anorm["anorm"]),
        "anorm_null_upper": float(anorm["null_upper"]),
        "bnorm":            float(bnorm["bnorm"]),
        "bnorm_null_lower": float(bnorm["null_lower"]),
    }


def _signal_gate_entangled(
    z_ap: np.ndarray,
    y_ap: np.ndarray,
    cfg: EvalConfig,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    splits = make_cv_splits(
        len(y_ap), y_ap, n_folds=cfg.n_folds,
        rng=np.random.default_rng(rng.integers(1 << 63)),
    )
    observed = cv_auroc(z_ap, y_ap, splits=splits)
    seeds = np.random.default_rng(rng.integers(1 << 63)).integers(
        1 << 63, size=cfg.n_permutations,
    )
    null_vals = np.array([
        cv_auroc(z_ap, np.random.default_rng(s).permutation(y_ap), splits=splits)
        for s in seeds
    ])
    p = (1 + np.sum(null_vals >= observed)) / (cfg.n_permutations + 1)
    return bool(p < 0.05), float(observed)


def _select_signal_component(
    ds: dict,
    z_shared: dict[str, np.ndarray],
    z_specific: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], str]:
    if is_modality_dominant(ds.get("ci_upper"), ds.get("delta_null_lower")):
        return z_specific, "modality"
    if is_shared_dominant(ds.get("ci_lower"), ds.get("delta_null_upper")):
        return z_shared, "shared"
    return z_shared, "indeterminate"


def _compute_stage4(
    z_ap: np.ndarray,
    y_ap: np.ndarray,
    z_b: np.ndarray,
    y_b: np.ndarray,
    z_c: np.ndarray,
    z_ap_raw: np.ndarray,
    z_b_raw: np.ndarray,
    z_c_raw: np.ndarray,
    cfg: EvalConfig,
    signal_present: bool,
    rng: np.random.Generator,
) -> dict:
    """Compute Stage IV cross-cohort stability metrics.

    Parameters
    ----------
    z_ap, z_b, z_c : ndarray
        Per-cohort standardised signal component — used for P_transfer.
    z_ap_raw, z_b_raw, z_c_raw : ndarray
        Raw (no per-cohort standardisation) signal component — used for
        D_quantile^task, which must preserve cohort mean shifts in the
        probe direction.
    signal_present : bool
        Passed through for logging/record purposes only.  D_quantile^task
        is now computed unconditionally: its own permutation null controls
        the FP rate, and gating it on signal_present blocked S2B detection
        when individual-level AUROC was marginal but cohort-level shift was
        strong (the original gate was designed for prototype-based D_quantile
        and is inappropriate for the probe-direction variant).
    """
    t_ci = compute_transfer_ci(
        z_ap, y_ap, z_b, y_b, z_c,
        n_bootstrap=cfg.n_bootstrap,
        rng=np.random.default_rng(rng.integers(1 << 63)),
    )
    t_null = compute_transfer_null(
        z_ap, y_ap, z_b, y_b, z_c,
        n_permutations=cfg.n_permutations,
        n_jobs=cfg.n_jobs,
        rng=np.random.default_rng(rng.integers(1 << 63)),
    )
    # D_quantile^task: primary Scenario 2 metric. Computed unconditionally
    # — the permutation null (shuffling B/C labels with fixed probe direction)
    # controls FP regardless of signal_present.  Sentinel zeros would
    # suppress cohort-level shift detection for S2B when individual-level
    # AUROC is marginal, creating a blind spot.
    dq_task = compute_dquantile_task(
        z_ap_raw, z_b_raw, z_c_raw, y_ap,
        n_permutations=cfg.n_permutations,
        rng=np.random.default_rng(rng.integers(1 << 63)),
        probe_backend=cfg.dq_probe_backend,
    )

    return {
        "p_transfer":                float(t_ci["p_transfer"]),
        "transfer_ci_lower":         float(t_ci["ci_lower"]),
        "transfer_ci_upper":         float(t_ci["ci_upper"]),
        "transfer_null_upper":       float(t_null["null_upper"]),
        "dquantile_task":            float(dq_task["dquantile_task"]),
        "dquantile_task_null_upper": float(dq_task["dquantile_task_null_upper"]),
    }


def _build_record(
    run_id: str,
    model_name: str,
    alpha_cfg: AlphaConfig,
    n_eval: int,
    modality: str,
    stage1: dict,
    ds: dict | None,
    signal_present: bool,
    localization: str | None,
    stage4: dict,
    scenario: str,
) -> dict:
    return {
        "run_id":             run_id,
        "model":              model_name,
        "alpha_config_id":    alpha_cfg.config_id,
        "alpha_s":            alpha_cfg.alpha_s,
        "alpha_h":            alpha_cfg.alpha_h,
        "alpha_r":            alpha_cfg.alpha_r,
        "alpha_b":            alpha_cfg.alpha_b,
        "n_eval":             n_eval,
        "modality":           modality,
        "predicted_scenario": scenario,
        "signal_present":     signal_present,
        "anorm":              stage1["anorm"],
        "anorm_null_upper":   stage1["anorm_null_upper"],
        "bnorm":              stage1["bnorm"],
        "bnorm_null_lower":   stage1["bnorm_null_lower"],
        "delta_shared":       ds["delta_shared"] if ds else None,
        "delta_ci_lower":     ds["ci_lower"] if ds else None,
        "delta_ci_upper":     ds["ci_upper"] if ds else None,
        "delta_null_lower":   ds.get("delta_null_lower") if ds else None,
        "delta_null_upper":   ds.get("delta_null_upper") if ds else None,
        "auroc_shared":       ds.get("auroc_shared") if ds else None,
        "auroc_specific":     ds.get("auroc_specific") if ds else None,
        "auroc_both":         ds.get("auroc_both") if ds else None,
        "localization":       localization,
        **stage4,
    }


def _serialize(record: dict) -> dict:
    """Convert numpy scalar types to Python-native for JSON serialization."""
    out = {}
    for k, v in record.items():
        if isinstance(v, np.bool_):
            out[k] = bool(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def evaluate_runs_parallel(
    run_ids: list[str],
    repr_dir: str | Path,
    alpha_configs: list,
    eval_config: EvalConfig,
    output_dir: str | Path,
    n_concurrent: int = 10,
) -> None:
    """Evaluate multiple runs concurrently using a process pool.

    Dispatches ``n_concurrent`` independent ``evaluate_run()`` calls at a time,
    each using ``eval_config.n_jobs`` threads internally.  On a 178-CPU machine,
    ``n_concurrent=10`` with ``eval_config.n_jobs=16`` utilises ~160 CPUs.

    Already-evaluated runs (output file exists) are skipped automatically.

    Parameters
    ----------
    run_ids : list of str
        Run identifiers, each matching a subdirectory in repr_dir.
    repr_dir : str or Path
        Root directory of saved representations.
    alpha_configs : list of AlphaConfig
        Outcome configurations to evaluate.
    eval_config : EvalConfig
        Evaluation hyperparameters (n_jobs controls per-run parallelism).
    output_dir : str or Path
        Directory to write per-run JSON result files.
    n_concurrent : int
        Number of evaluate_run() calls to run simultaneously.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        rid for rid in run_ids
        if not (output_dir / f"{rid}.json").exists()
    ]
    if not pending:
        print("All runs already evaluated.", flush=True)
        return

    print(
        f"Evaluating {len(pending)}/{len(run_ids)} runs "
        f"({len(run_ids) - len(pending)} already done), "
        f"n_concurrent={n_concurrent}, n_jobs={eval_config.n_jobs}",
        flush=True,
    )

    t0_parallel = time.time()
    n_total = len(pending)

    def _worker(run_id: str) -> str:
        # Prevent BLAS/MKL from spawning extra threads inside each worker
        # process — without this, n_concurrent loky processes each spawning
        # OMP_NUM_THREADS threads can multiply total thread count beyond
        # the physical core count.
        import os as _os
        _os.environ.setdefault("OMP_NUM_THREADS", "2")
        _os.environ.setdefault("MKL_NUM_THREADS", "2")
        _os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        evaluate_run(
            run_id=run_id,
            repr_dir=str(repr_dir),
            alpha_configs=alpha_configs,
            eval_config=eval_config,
            output_dir=str(output_dir),
            seed=0,
        )
        return run_id

    for i, run_id in enumerate(
        Parallel(n_jobs=n_concurrent, backend="loky", return_as="generator")(
            delayed(_worker)(rid) for rid in pending
        ),
        start=1,
    ):
        elapsed = time.time() - t0_parallel
        rate = i / elapsed
        eta = (n_total - i) / rate if rate > 0 else float("inf")
        print(
            f"  [{i}/{n_total}] {run_id} done  "
            f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
            flush=True,
        )


def main() -> None:
    """CLI entrypoint: evaluate all scenarios for one synthetic run."""
    parser = argparse.ArgumentParser(description="DECAT scenario evaluation")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repr-dir", default=f"{DEFAULT_RESULTS}/representations")
    parser.add_argument("--output-dir", default=f"{DEFAULT_RESULTS}/evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    cfg = EvalConfig(
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        n_jobs=args.n_jobs,
    )
    evaluate_run(
        run_id=args.run_id,
        repr_dir=args.repr_dir,
        alpha_configs=make_alpha_configs(),
        eval_config=cfg,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
