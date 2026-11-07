"""TCGA Validation — DECAT Evaluation.

Evaluates all (split, model, task, modality) combinations:

  Pooled random splits:
    S1 labels (CCA variates 0-K)  × models × modalities
    S4 labels (JIVE RNA PCs 0-K)  × models × modalities
    S3 labels (random 0-9)        × models × modalities
    S2B labels (cca_all variates) × models × modalities

  Extreme-C splits (CCA variate S2B evaluation):
    cca_all extreme-C splits × models × modalities
    A'/B matched pan-cancer; C drawn from label-extreme pool for S2 detection.

Workers load representations lazily from disk (no large array serialisation).

Results::

    <OUT_DIR>/results/pooled_random_split<N>.json
    <OUT_DIR>/results/cca_all_extreme_c_split<N>.json

Usage::

    python run_decat.py --out-dir <dir>
    python run_decat.py --out-dir <dir> --seed-only 0   # quick sanity check
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import os
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parents[2]))

OUT_DIR     = os.environ.get('DECAT_OUT_DIR', './data/tcga')  # override with --out-dir
N_PERM      = 200
N_BOOTSTRAP = 200
K_CV        = 5
N_JOBS      = max(1, os.cpu_count() - 2)  # use all but 2 CPUs


def ts(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _combo_seed(split_id, split_type, model_name, task_type, task_idx) -> int:
    key = f"{split_id}:{split_type}:{model_name}:{task_type}:{task_idx}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (1 << 63)


def _std(Z: np.ndarray) -> np.ndarray:
    return (Z - Z.mean(0)) / Z.std(0).clip(min=1e-8)


def _safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _serialize(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, np.bool_):
            out[k] = bool(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _run_stage1(zh_ap, zr_ap, rng):
    from metrics.anorm import compute_anorm_null
    result = compute_anorm_null(zh_ap, zr_ap, n_permutations=N_PERM, rng=rng)
    return {"anorm": float(result["anorm"]), "anorm_null_upper": float(result["null_upper"])}


def _signal_gate_entangled(Z_ap, y, rng):
    from evaluation.probes import cv_auroc, make_cv_splits
    splits   = make_cv_splits(len(y), y, n_folds=K_CV, rng=rng)
    observed = cv_auroc(Z_ap, y, splits=splits)
    seeds    = np.random.default_rng(rng.integers(1 << 63)).integers(1 << 63, size=N_PERM)
    null_vals = np.array([
        cv_auroc(Z_ap, np.random.default_rng(s).permutation(y), splits=splits)
        for s in seeds
    ])
    p = (1 + np.sum(null_vals >= observed)) / (N_PERM + 1)
    return bool(p < 0.05)


def _run_stage3(Z_shared_ap, Z_specific_ap, y, rng):
    from metrics.delta_shared import compute_delta_shared_bootstrap
    result = compute_delta_shared_bootstrap(
        Z_shared_ap, Z_specific_ap, y,
        n_folds=K_CV, n_bootstrap=N_BOOTSTRAP,
        n_permutations=N_PERM, n_jobs=1, rng=rng, standardize=False,
    )
    return {
        "delta_shared":     _safe_float(result.get("delta_shared")),
        "delta_ci_lower":   _safe_float(result.get("ci_lower")),
        "delta_ci_upper":   _safe_float(result.get("ci_upper")),
        "delta_null_lower": _safe_float(result.get("delta_null_lower")),
        "delta_null_upper": _safe_float(result.get("delta_null_upper")),
        "auroc_shared":     _safe_float(result.get("auroc_shared")),
        "auroc_specific":   _safe_float(result.get("auroc_specific")),
        "signal_present":   bool(result.get("signal_present", False)),
    }


def _compute_cohort_aurocs(Z_ap, y_ap, Z_b, y_b, Z_c, y_c, rng):
    """CV AUROC on A', held-out AUROC on B and C using A'-trained probe.

    B is the natural unbiased out-of-sample estimate on a balanced cohort.
    C is the deployment cohort — may be inflated for label-extreme splits.
    Returns (auroc_ap, auroc_b, auroc_c); any value is None if not computable.
    """
    from evaluation.probes import cv_auroc, make_cv_splits
    from sklearn.metrics import roc_auc_score

    def _usable(y):
        valid = y[y >= 0]
        return len(valid) >= 10 and len(np.unique(valid)) > 1

    if not _usable(y_ap):
        return None, None, None

    splits   = make_cv_splits(len(y_ap), y_ap, n_folds=K_CV, rng=rng)
    auroc_ap = float(cv_auroc(Z_ap, y_ap, splits=splits))

    try:
        from evaluation.probes import fit_probe, predict_scores
        _, probe = fit_probe(Z_ap, y_ap, probe_backend='logistic',
                             C=0.1, max_iter=500, random_state=0, standardize=False)
        mask_b = y_b >= 0
        auroc_b = (float(roc_auc_score(y_b[mask_b], predict_scores(None, probe, Z_b[mask_b])))
                   if _usable(y_b) else None)
        mask_c = y_c >= 0
        auroc_c = (float(roc_auc_score(y_c[mask_c], predict_scores(None, probe, Z_c[mask_c])))
                   if _usable(y_c) else None)
    except Exception:
        auroc_b, auroc_c = None, None

    return auroc_ap, auroc_b, auroc_c


def _run_stage4(Z_ap, y_ap, Z_b, y_b, Z_c, Z_ap_raw, Z_b_raw, Z_c_raw, rng):
    from metrics.transfer import compute_transfer_ci, compute_transfer_null
    from metrics.quantile import compute_dquantile_task
    from evaluation.probes import fit_probe, predict_scores
    t_ci   = compute_transfer_ci(Z_ap, y_ap, Z_b, y_b, Z_c, n_bootstrap=N_BOOTSTRAP,
                                 rng=np.random.default_rng(rng.integers(1 << 63)))
    t_null = compute_transfer_null(Z_ap, y_ap, Z_b, y_b, Z_c, n_permutations=N_PERM,
                                   rng=np.random.default_rng(rng.integers(1 << 63)))
    dq     = compute_dquantile_task(Z_ap_raw, Z_b_raw, Z_c_raw, y_ap,
                                    n_permutations=N_PERM,
                                    rng=np.random.default_rng(rng.integers(1 << 63)))
    return {
        "p_transfer":                float(t_ci["p_transfer"]),
        "transfer_ci_lower":         float(t_ci["ci_lower"]),
        "transfer_ci_upper":         float(t_ci["ci_upper"]),
        "transfer_null_upper":       float(t_null["null_upper"]),
        "dquantile_task":            float(dq["dquantile_task"]),
        "dquantile_task_null_upper": float(dq["dquantile_task_null_upper"]),
    }


def _select_signal_component(ds3, z_shared, z_specific):
    from decision.scenario_classifier import is_shared_dominant, is_modality_dominant
    if is_modality_dominant(ds3.get("delta_ci_upper"), ds3.get("delta_null_lower")):
        return z_specific, "modality"
    if is_shared_dominant(ds3.get("delta_ci_lower"), ds3.get("delta_null_upper")):
        return z_shared, "shared"
    return z_shared, "indeterminate"


def _evaluate_one(
    split_id, split_type, model_name, is_factorized, gamma,
    Zh, Zr, y_ap, y_b, y_c, task_type, task_idx, seed,
) -> list[dict]:
    from metrics.delta_shared import soft_mask_representations
    from decision.scenario_classifier import MetricBundle, classify_scenario

    # Filter out patients with missing labels (-1 sentinel for age/tp53/tmb).
    # A' and B are trimmed to valid-label patients only; C is unaffected
    # (never used for label-dependent computation).
    mask_ap = y_ap >= 0
    mask_b  = y_b  >= 0
    if not mask_ap.all():
        Zh = {k: (Zh[k][mask_ap] if k == "Ap" else Zh[k]) for k in Zh}
        Zr = {k: (Zr[k][mask_ap] if k == "Ap" else Zr[k]) for k in Zr}
        y_ap = y_ap[mask_ap]
    if not mask_b.all():
        Zh = {k: (Zh[k][mask_b] if k == "B" else Zh[k]) for k in Zh}
        Zr = {k: (Zr[k][mask_b] if k == "B" else Zr[k]) for k in Zr}
        y_b = y_b[mask_b]
    if len(y_ap) < 10 or len(y_b) < 10:
        return []  # too few valid-label patients for reliable evaluation

    rng       = np.random.default_rng(seed)
    zh_ap_std = _std(Zh["Ap"])
    zr_ap_std = _std(Zr["Ap"])
    stage1    = _run_stage1(zh_ap_std, zr_ap_std,
                             rng=np.random.default_rng(rng.integers(1 << 63)))

    records = []
    for modality in ("h", "r"):
        rng_mod  = np.random.default_rng(rng.integers(1 << 63))
        Z_raw    = Zh if modality == "h" else Zr
        Z_ap_raw = Z_raw["Ap"]
        Z_b_raw  = Z_raw["B"]
        Z_c_raw  = Z_raw["C"]
        Z_ap_std = _std(Z_ap_raw)
        Z_b_std  = _std(Z_b_raw)
        Z_c_std  = _std(Z_c_raw)

        if is_factorized and gamma is not None:
            Z_sh_ap_raw, Z_sp_ap_raw = soft_mask_representations(Z_ap_raw, gamma)
            Z_sh_ap,     Z_sp_ap     = soft_mask_representations(Z_ap_std, gamma)
            Z_sh_b_raw,  Z_sp_b_raw  = soft_mask_representations(Z_b_raw,  gamma)
            Z_sh_b,      Z_sp_b      = soft_mask_representations(Z_b_std,  gamma)
            Z_sh_c_raw,  Z_sp_c_raw  = soft_mask_representations(Z_c_raw,  gamma)
            Z_sh_c,      Z_sp_c      = soft_mask_representations(Z_c_std,  gamma)

            ds3 = _run_stage3(Z_sh_ap, Z_sp_ap, y_ap,
                              rng=np.random.default_rng(rng_mod.integers(1 << 63)))
            signal_present = ds3["signal_present"]

            z_sh      = {"Ap": Z_sh_ap,     "B": Z_sh_b,     "C": Z_sh_c}
            z_sp      = {"Ap": Z_sp_ap,     "B": Z_sp_b,     "C": Z_sp_c}
            z_sh_raw  = {"Ap": Z_sh_ap_raw, "B": Z_sh_b_raw, "C": Z_sh_c_raw}
            z_sp_raw  = {"Ap": Z_sp_ap_raw, "B": Z_sp_b_raw, "C": Z_sp_c_raw}

            z_sig,     localization = _select_signal_component(ds3, z_sh, z_sp)
            z_sig_raw, _            = _select_signal_component(ds3, z_sh_raw, z_sp_raw)
        else:
            ds3            = None
            localization   = "non_factorized"
            signal_present = _signal_gate_entangled(
                Z_ap_std, y_ap, rng=np.random.default_rng(rng_mod.integers(1 << 63)))
            z_sig     = {"Ap": Z_ap_std, "B": Z_b_std, "C": Z_c_std}
            z_sig_raw = {"Ap": Z_ap_raw, "B": Z_b_raw, "C": Z_c_raw}

        stage4 = _run_stage4(
            z_sig["Ap"], y_ap, z_sig["B"], y_b, z_sig["C"],
            z_sig_raw["Ap"], z_sig_raw["B"], z_sig_raw["C"],
            rng=np.random.default_rng(rng_mod.integers(1 << 63)),
        )

        auroc_ap, auroc_b, auroc_c = _compute_cohort_aurocs(
            z_sig["Ap"], y_ap,
            z_sig["B"],  y_b,
            z_sig["C"],  y_c,
            rng=np.random.default_rng(rng_mod.integers(1 << 63)),
        )

        bundle = MetricBundle(
            anorm=stage1["anorm"],
            anorm_null_upper=stage1["anorm_null_upper"],
            signal_present=signal_present,
            transfer_ci_lower=stage4["transfer_ci_lower"],
            transfer_null_upper=stage4["transfer_null_upper"],
            dquantile_task=stage4["dquantile_task"],
            dquantile_task_null_upper=stage4["dquantile_task_null_upper"],
            is_factorized=bool(is_factorized),
            delta_ci_lower=ds3["delta_ci_lower"]   if ds3 else None,
            delta_ci_upper=ds3["delta_ci_upper"]   if ds3 else None,
            delta_null_lower=ds3["delta_null_lower"] if ds3 else None,
            delta_null_upper=ds3["delta_null_upper"] if ds3 else None,
        )
        predicted = classify_scenario(bundle)

        records.append({
            "split_id":               split_id,
            "split_type":             split_type,
            "model":                  model_name,
            "modality":               modality,
            "task_type":              task_type,
            "task_idx":               task_idx,
            "predicted_scenario":     predicted,
            "signal_present":         signal_present,
            "localization":           localization,
            "anorm":                  stage1["anorm"],
            "anorm_null_upper":       stage1["anorm_null_upper"],
            "delta_shared":           ds3["delta_shared"]     if ds3 else None,
            "delta_ci_lower":         ds3["delta_ci_lower"]   if ds3 else None,
            "delta_ci_upper":         ds3["delta_ci_upper"]   if ds3 else None,
            "delta_null_lower":       ds3["delta_null_lower"] if ds3 else None,
            "delta_null_upper":       ds3["delta_null_upper"] if ds3 else None,
            **stage4,
            "auroc_ap":               auroc_ap,
            "auroc_b":                auroc_b,
            "auroc_c":                auroc_c,
        })

    return records


def _safe_evaluate_one(*args, **kwargs) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS",     "2")
    os.environ.setdefault("MKL_NUM_THREADS",     "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
    try:
        return _evaluate_one(*args, **kwargs)
    except Exception as e:
        ts(f"  ERROR: {e}\n{traceback.format_exc()}")
        return []


def _load_cohorts(repr_dir, model_name, idx_ap, idx_b, idx_c):
    """Load full_latents.npz and index into cohort subsets.
    joblib loky backend mmaps large numpy arrays automatically — no copy."""
    full  = np.load(repr_dir / model_name / "full_latents.npz")
    Zh_all, Zr_all = full["Zh"], full["Zr"]
    Zh    = {"Ap": Zh_all[idx_ap], "B": Zh_all[idx_b], "C": Zh_all[idx_c]}
    Zr    = {"Ap": Zr_all[idx_ap], "B": Zr_all[idx_b], "C": Zr_all[idx_c]}
    gpath = repr_dir / model_name / "gamma.npy"
    gamma = np.load(gpath) if gpath.exists() else None
    return Zh, Zr, gamma


def _build_all_jobs(
    split, sid, split_type, model_names, repr_dir,
    s1_labels, s4_labels, s3_labels, s2b_labels,
    vi=None,
):
    """Build jobs for one split type (pooled or extreme-C).

    sid and split_type are passed explicitly so this works for all three
    split structures (pooled uses 'seed', others use 'split_id').

    vi: variate_indices list — maps position k (0..K-1) to actual variate
    index (e.g. 0,5,10,...,45).  Used as task_idx in records so figure
    x-axes show the real variate number.  Falls back to sequential 0..K-1.
    """
    idx_ap = np.array(split["cohort_aprime"])
    idx_b  = np.array(split["cohort_b"])
    idx_c  = np.array(split["cohort_c"])
    jobs   = []

    def _y(row):
        return (row[idx_ap].astype(np.int8),
                row[idx_b].astype(np.int8),
                row[idx_c].astype(np.int8))

    for mn in model_names:
        Zh, Zr, gamma = _load_cohorts(repr_dir, mn, idx_ap, idx_b, idx_c)
        is_fac = gamma is not None
        base   = (sid, split_type, mn, is_fac, gamma, Zh, Zr)

        for k in range(s1_labels.shape[0]):
            vi_k = vi[k] if vi is not None else k   # actual variate index
            y_ap, y_b, y_c = _y(s1_labels[k])
            jobs.append((*base, y_ap, y_b, y_c, "s1", vi_k,
                         _combo_seed(sid, split_type, mn, "s1", vi_k)))
        for k in range(s4_labels.shape[0]):
            vi_k = vi[k] if vi is not None else k
            y_ap, y_b, y_c = _y(s4_labels[k])
            jobs.append((*base, y_ap, y_b, y_c, "s4", vi_k,
                         _combo_seed(sid, split_type, mn, "s4", vi_k)))
        for k in range(s3_labels.shape[0]):
            y_ap, y_b, y_c = _y(s3_labels[k])
            jobs.append((*base, y_ap, y_b, y_c, "s3", k,
                         _combo_seed(sid, split_type, mn, "s3", k)))
        for k in range(s2b_labels.shape[0]):
            # For s2b_cca (first vi entries), use actual variate index;
            # for s2b_age/tp53/tmb (single labels after cca), use k as-is
            vi_k = vi[k] if (vi is not None and k < len(vi)) else k
            y_ap, y_b, y_c = _y(s2b_labels[k])
            jobs.append((*base, y_ap, y_b, y_c, "s2b", vi_k,
                         _combo_seed(sid, split_type, mn, "s2b", vi_k)))

    return jobs



def print_summary(res_dir: Path) -> None:
    import glob

    all_records = []
    for f in glob.glob(str(res_dir / "*.json")):
        all_records.extend(json.load(open(f)))

    if not all_records:
        ts("No records found for summary.")
        return

    def rate(recs, pred):
        return sum(r["predicted_scenario"] == pred for r in recs) / len(recs) if recs else float("nan")

    models = sorted({r["model"] for r in all_records})
    PRED   = {"s1": "scenario_1", "s4": "scenario_4", "s3": "scenario_3", "s2b": "scenario_2"}

    n_patients = len({r.get("split_id") for r in all_records})  # proxy
    print("\n" + "="*70)
    print(f"SUMMARY: DECAT TCGA validation  ({len(all_records):,} records)")
    print("="*70)

    # 1. Pooled splits: S1/S4/S3 detection + FSCR
    pooled = [r for r in all_records if r.get("split_type") == "Pooled random"
              or "pooled" in str(r.get("split_type","")).lower()]
    if pooled:
        print("\n1. Pooled splits — detection rates by model\n")
        print(f"  {'Model':<26}  {'S1→S1':>7}  {'S4→S4':>7}  {'S3→S3':>7}  {'FSCR':>7}")
        print("  " + "-"*55)
        for mn in models:
            m = [r for r in pooled if r["model"] == mn]
            s1  = rate([r for r in m if r["task_type"] == "s1"], PRED["s1"])
            s4  = rate([r for r in m if r["task_type"] == "s4"], PRED["s4"])
            s3  = rate([r for r in m if r["task_type"] == "s3"], PRED["s3"])
            s4r = [r for r in m if r["task_type"] == "s4"]
            fscr_v = rate([r for r in s4r], "scenario_1")
            print(f"  {mn:<26}  {s1:>7.3f}  {s4:>7.3f}  {s3:>7.3f}  {fscr_v:>7.3f}")

    # 2. Extreme-C splits: S2B detection by label type
    ec_types = [("cca_all_extreme_c", "CCA extreme-C")]
    ec_found = [(st, label) for st, label in ec_types
                if any(st in str(r.get("split_type", "")) for r in all_records)]
    if ec_found:
        print("\n2. Extreme-C splits — S2 detection rate (P_transfer + D_quantile both fire)\n")
        print(f"  {'Model':<26}  {'Mod':>3}  " +
              "  ".join(f"{label[:12]:>12}" for _, label in ec_found))
        print("  " + "-"*(30 + 15*len(ec_found)))
        for mn in models:
            for mod in ("h", "r"):
                row = f"  {mn:<26}  {mod:>3}"
                for st, _ in ec_found:
                    recs = [r for r in all_records
                            if r["model"] == mn and r["modality"] == mod
                            and st in str(r.get("split_type", ""))]
                    row += f"  {rate(recs, PRED['s2b']):>12.3f}"
                print(row)
            print()

    # 3. S1 detection vs variate index (pooled)
    s1_recs = [r for r in pooled if r["task_type"] == "s1"] if pooled else []
    if s1_recs:
        max_k = max(r["task_idx"] for r in s1_recs)
        print(f"\n3. S1 detection vs variate index (pooled, all models/modalities)\n")
        for k in sorted({r["task_idx"] for r in s1_recs}):
            recs = [r for r in s1_recs if r["task_idx"] == k]
            print(f"  variate {k:2d}: {rate(recs, PRED['s1']):.3f}  (n={len(recs)})")

    print("="*70 + "\n")


def _run_splits_impl(split_list, split_type, out_prefix, get_sid,
                     s1, s4, s3, s2b,
                     res_dir, model_names, repr_dir,
                     seed_only=None, vi=None):
    """Run all tasks on one set of splits, saving one file per split."""
    if seed_only is not None:
        split_list = [s for s in split_list if get_sid(s) == seed_only]
    ts(f"{split_type}: {len(split_list)} splits × {len(model_names)} models")
    for split in split_list:
        sid      = get_sid(split)
        out_path = res_dir / f"{out_prefix}{sid}.json"
        if out_path.exists():
            ts(f"  split {sid} already done, skipping")
            continue
        jobs = _build_all_jobs(split, sid, split_type, model_names, repr_dir, s1, s4, s3, s2b, vi)
        n_ap = len(split.get('cohort_aprime', []))
        n_b  = len(split.get('cohort_b', []))
        n_c  = len(split.get('cohort_c', []))
        ts(f"  split {sid}: {len(jobs)} jobs (A'={n_ap}, B={n_b}, C={n_c})")
        t0      = time.time()
        batches = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_safe_evaluate_one)(*a) for a in jobs)
        records = [r for b in batches for r in b]
        ts(f"  done in {time.time()-t0:.1f}s — {len(records)} records")
        with open(out_path, "w") as fh:
            json.dump([_serialize(r) for r in records], fh)


def main() -> None:
    import argparse as _ap2
    _p = _ap2.ArgumentParser()
    _p.add_argument("--he-csv",    default=None)
    _p.add_argument("--out-dir",   default=None)
    _p.add_argument("--seed-only", type=int, default=None,
                    help="Run only this split index (0-based) for quick sanity check")
    args = _p.parse_args()

    ts("=== TCGA Validation — DECAT Evaluation ===")

    out_dir  = Path(args.out_dir if args.out_dir else OUT_DIR)
    repr_dir = out_dir / "representations"
    res_dir  = out_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    ts(f"N_JOBS = {N_JOBS} (CPUs: {os.cpu_count()})")

    ts("Loading splits and task labels ...")
    with open(out_dir / "pooled_random_splits.json") as f:
        pooled_splits = json.load(f)


    task_data       = np.load(out_dir / "task_labels.npz")
    s1_labels       = task_data["s1_labels"]   # (K_v, N)
    s4_labels       = task_data["s4_labels"]   # (K_v, N)
    s3_labels       = task_data["s3_labels"]   # (10, N)
    # Variate indices saved by prepare_data.py (currently [0,1,...,49]).
    # Used as task_idx so figures show the real variate number.
    # Falls back to sequential 0..K-1 for old results without this field.
    variate_indices = (task_data["variate_indices"].tolist()
                       if "variate_indices" in task_data
                       else list(range(s1_labels.shape[0])))

    # S2B labels — all three variants
    _N = s1_labels.shape[1]
    s2b_cca  = task_data["s2b_cca"]  if "s2b_cca"  in task_data else (
               task_data["s2b_labels"][np.newaxis, :] if "s2b_labels" in task_data
               else np.zeros((1, _N), dtype=np.int8))
    s2b_age  = task_data["s2b_age"]  if "s2b_age"  in task_data else np.full(_N, -1, dtype=np.int8)
    s2b_tp53 = task_data["s2b_tp53"] if "s2b_tp53" in task_data else np.full(_N, -1, dtype=np.int8)
    s2b_tmb  = task_data["s2b_tmb"]  if "s2b_tmb"  in task_data else np.full(_N, -1, dtype=np.int8)

    # Promote 1D labels (N,) → (1, N) to match s2b_cca shape (K_v, N) for concatenation
    if s2b_age.ndim  == 1: s2b_age  = s2b_age[np.newaxis, :]
    if s2b_tp53.ndim == 1: s2b_tp53 = s2b_tp53[np.newaxis, :]
    if s2b_tmb.ndim  == 1: s2b_tmb  = s2b_tmb[np.newaxis, :]

    # Combined S2B: cca sweep + age + tp53 + tmb
    s2b_all = np.concatenate([s2b_cca, s2b_age, s2b_tp53, s2b_tmb], axis=0)
    n_cca = s2b_cca.shape[0]; n_age = s2b_age.shape[0]
    n_tp53 = s2b_tp53.shape[0]; n_tmb = s2b_tmb.shape[0]
    ts(f"  S2B labels: cca={n_cca}, age={n_age}, tp53={n_tp53}, tmb={n_tmb} (total={s2b_all.shape[0]})")

    model_names = sorted(d.name for d in repr_dir.iterdir() if d.is_dir())
    ts(f"Models: {model_names}")
    ts(f"S1: {s1_labels.shape}, S4: {s4_labels.shape}, S3: {s3_labels.shape}, S2B_all: {s2b_all.shape}")

    def _run_splits(split_list, split_type, out_prefix, get_sid,
                    s1, s4, s3, s2b, vi=None):
        _run_splits_impl(split_list, split_type, out_prefix, get_sid,
                         s1, s4, s3, s2b,
                         res_dir=res_dir, model_names=model_names,
                         repr_dir=repr_dir, seed_only=args.seed_only,
                         vi=vi)

    # Add S2B to pooled jobs builder — reuse _build_pooled_jobs which handles it
    _run_splits(pooled_splits, "Pooled random", "pooled_random_split",
                lambda s: s["seed"], s1_labels, s4_labels, s3_labels, s2b_all,
                vi=variate_indices)

    # Only s2b is evaluated per label (s1/s4/s3 use empty arrays so no jobs
    # are generated for those task types).
    _empty = np.zeros((0, _N), dtype=np.int8)
    _ec_labels = [("cca_all", s2b_cca),   # all 50 CCA variates — S1→S2 sweep
                  ]
    for _lt, _label in _ec_labels:
        _path = out_dir / f"{_lt}_extreme_c_splits.json"
        if _path.exists():
            _splits = json.load(open(_path))
            ts(f"{_lt.upper()} extreme-C: {len(_splits)} splits loaded")
            _run_splits(_splits, f"{_lt}_extreme_c", f"{_lt}_extreme_c_split",
                        lambda s: s["split_id"], _empty, _empty, _empty, _label)
        else:
            ts(f"  {_lt}_extreme_c_splits.json not found — skipping")

    ts("=== Step 4 complete ===")
    print_summary(res_dir)


if __name__ == "__main__":
    main()