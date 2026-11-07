"""Scenario identifiability metrics for DECAT experiments.

All functions operate on a list of record dicts (or a pandas DataFrame)
with at least ``ground_truth`` and ``predicted_scenario`` columns.
Optional filters: ``n_eval``, ``model``, ``modality``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

Records = list[dict] | pd.DataFrame

SCENARIOS = ["scenario_1", "scenario_2", "scenario_3", "scenario_4"]
SCENARIO_ORDER = SCENARIOS + ["indeterminate"]


def _to_df(records: Records) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    return pd.DataFrame(records)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    if n == 0:
        return 0.0, 0.0
    lo = float(beta_dist.ppf(alpha / 2, max(k, 1e-9), max(n - k, 1e-9) + 1))
    hi = float(beta_dist.ppf(1 - alpha / 2, k + 1, max(n - k, 1e-9)))
    return lo, hi


def strict_accuracy(records: Records, **filters) -> float:
    """Fraction where ground truth matches predicted scenario.

    Indeterminate predictions count as incorrect.

    Parameters
    ----------
    records : list[dict] or DataFrame
        Evaluation records with ``ground_truth`` and ``predicted_scenario``.
    **filters
        Column=value pairs passed to the internal filter (e.g. ``n_eval=500``).

    Returns
    -------
    float
        Strict accuracy in [0, 1], or NaN if no records match.
    """
    df = _filter(_to_df(records), **filters)
    if len(df) == 0:
        return float("nan")
    return float((df["ground_truth"] == df["predicted_scenario"]).mean())


def conservative_accuracy(records: Records, **filters) -> float:
    """Like strict, but indeterminate predictions are not penalized.

    Parameters
    ----------
    records : list[dict] or DataFrame
        Evaluation records with ``ground_truth`` and ``predicted_scenario``.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    float
        Conservative accuracy in [0, 1], or NaN if no records match.
    """
    df = _filter(_to_df(records), **filters)
    if len(df) == 0:
        return float("nan")
    correct_or_indet = (
        (df["ground_truth"] == df["predicted_scenario"])
        | (df["predicted_scenario"] == "indeterminate")
    )
    return float(correct_or_indet.mean())


def resolution_rate(records: Records, **filters) -> float:
    """Fraction of predictions that are non-indeterminate.

    Parameters
    ----------
    records : list[dict] or DataFrame
        Evaluation records with ``predicted_scenario``.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    float
        Resolution rate in [0, 1], or NaN if no records match.
    """
    df = _filter(_to_df(records), **filters)
    if len(df) == 0:
        return float("nan")
    return float((df["predicted_scenario"] != "indeterminate").mean())


def false_shared_claim_rate(records: Records, **filters) -> float:
    """FSCR: P(predicted=1 | true != 1). Shared-biology hallucination rate.

    Parameters
    ----------
    records : list[dict] or DataFrame
        Evaluation records with ``ground_truth`` and ``predicted_scenario``.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    float
        False shared claim rate in [0, 1], or NaN if no non-S1 records.
    """
    df = _filter(_to_df(records), **filters)
    non_s1 = df[df["ground_truth"] != "scenario_1"]
    if len(non_s1) == 0:
        return float("nan")
    return float((non_s1["predicted_scenario"] == "scenario_1").mean())


def false_confidence_rate(records: Records, **filters) -> float:
    """P(predicted != true and predicted != indeterminate). Confident but wrong.

    Parameters
    ----------
    records : list[dict] or DataFrame
        Evaluation records with ``ground_truth`` and ``predicted_scenario``.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    float
        False confidence rate in [0, 1], or NaN if no records match.
    """
    df = _filter(_to_df(records), **filters)
    if len(df) == 0:
        return float("nan")
    wrong_and_resolved = (
        (df["ground_truth"] != df["predicted_scenario"])
        & (df["predicted_scenario"] != "indeterminate")
    )
    return float(wrong_and_resolved.mean())


def confusion_matrix(records: Records, **filters) -> pd.DataFrame:
    """Row-normalized confusion matrix P(predicted=j | true=k).

    Parameters
    ----------
    records : list[dict] or DataFrame
        Evaluation records with ``ground_truth`` and ``predicted_scenario``.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    pd.DataFrame
        DataFrame with ground-truth rows and predicted columns, normalized
        so each row sums to 1.
    """
    df = _filter(_to_df(records), **filters)
    ct = pd.crosstab(
        df["ground_truth"], df["predicted_scenario"],
        dropna=False,
    )
    for s in SCENARIO_ORDER:
        if s not in ct.columns:
            ct[s] = 0
        if s not in ct.index:
            ct.loc[s] = 0
    ct = ct.reindex(index=SCENARIOS, columns=SCENARIO_ORDER, fill_value=0)
    row_sums = ct.sum(axis=1).replace(0, 1)
    return ct.div(row_sums, axis=0)


def confusion_counts(records: Records, **filters) -> pd.DataFrame:
    """Raw count confusion matrix (not normalized).

    Parameters
    ----------
    records : list[dict] or DataFrame
        Evaluation records with ``ground_truth`` and ``predicted_scenario``.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    pd.DataFrame
        DataFrame with ground-truth rows and predicted columns (raw counts).
    """
    df = _filter(_to_df(records), **filters)
    ct = pd.crosstab(
        df["ground_truth"], df["predicted_scenario"],
        dropna=False,
    )
    for s in SCENARIO_ORDER:
        if s not in ct.columns:
            ct[s] = 0
        if s not in ct.index:
            ct.loc[s] = 0
    return ct.reindex(index=SCENARIOS, columns=SCENARIO_ORDER, fill_value=0)


def scenario_rate(
    records: Records,
    ground_truth: str,
    predicted: str,
    **filters,
) -> dict:
    """P(predicted | true=ground_truth) with Clopper-Pearson CI.

    Parameters
    ----------
    records : list[dict] or DataFrame
        Evaluation records with ``ground_truth`` and ``predicted_scenario``.
    ground_truth : str
        Ground-truth scenario to condition on.
    predicted : str
        Predicted scenario to count.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    dict
        Keys: ``rate``, ``ci_lower``, ``ci_upper``, ``k``, ``n``.
    """
    df = _filter(_to_df(records), **filters)
    sub = df[df["ground_truth"] == ground_truth]
    n = len(sub)
    k = int((sub["predicted_scenario"] == predicted).sum())
    lo, hi = clopper_pearson(k, n)
    return {
        "rate": k / n if n > 0 else float("nan"),
        "ci_lower": lo,
        "ci_upper": hi,
        "k": k,
        "n": n,
    }


def power_curve(
    records: Records,
    ground_truth: str,
    predicted: str,
    n_eval_sizes: list[int] | None = None,
    **filters,
) -> pd.DataFrame:
    """Detection / FP rate as a function of N_eval.

    Parameters
    ----------
    records : list[dict] or DataFrame
        Evaluation records.
    ground_truth : str
        Ground-truth scenario to condition on.
    predicted : str
        Predicted scenario to count.
    n_eval_sizes : list[int] or None
        N_eval values to sweep. If None, uses all unique values in records.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    pd.DataFrame
        Columns: ``n_eval``, ``rate``, ``ci_lower``, ``ci_upper``, ``k``, ``n``.
    """
    df = _filter(_to_df(records), **filters)
    if n_eval_sizes is None:
        n_eval_sizes = sorted(df["n_eval"].unique())
    rows = []
    for ne in n_eval_sizes:
        r = scenario_rate(df, ground_truth, predicted, n_eval=ne)
        r["n_eval"] = ne
        rows.append(r)
    return pd.DataFrame(rows)


def mixed_scenario_resolution_accuracy(
    records: Records,
    **filters,
) -> dict:
    """Fraction of runs where both modalities are correctly assigned when their
    ground-truth scenarios differ (spec §7 Mixed-scenario resolution accuracy).

    Only applicable to (run, model, alpha_config, n_eval) tuples where H&E and
    RNA have different ground-truth scenarios — in practice S4h (H&E=S4, RNA=S3)
    and S4r (H&E=S3, RNA=S4).  For each such tuple, both modalities must be
    correctly assigned for the run to count as resolved.

    Parameters
    ----------
    records : list[dict] or DataFrame
        Must contain ``run_id``, ``model``, ``alpha_config_id``, ``n_eval``,
        ``modality``, ``ground_truth``, ``predicted_scenario``.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    dict
        Keys: ``rate``, ``ci_lower``, ``ci_upper``, ``k``, ``n``.
    """
    df = _filter(_to_df(records), **filters)

    # Keep only rows where modalities differ in ground truth (S4h and S4r)
    mixed_prefixes = ("s4h", "s4r")
    df_mixed = df[df["alpha_config_id"].str.startswith(mixed_prefixes, na=False)].copy()

    if len(df_mixed) == 0:
        return {"rate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "k": 0, "n": 0}

    # Group by (run_id, model, alpha_config_id, n_eval) — one H&E row + one RNA row per group
    group_keys = ["run_id", "model", "alpha_config_id", "n_eval"]
    # Ensure required columns exist; fall back to model_label if model missing
    for key in group_keys:
        if key not in df_mixed.columns and key == "model" and "model_label" in df_mixed.columns:
            df_mixed = df_mixed.rename(columns={"model_label": "model"})

    group_keys = [k for k in group_keys if k in df_mixed.columns]
    groups = df_mixed.groupby(group_keys)

    n_total = 0
    n_correct = 0
    for _, grp in groups:
        if len(grp) != 2:
            continue  # need exactly both modalities
        # Both must be correctly assigned
        correct = (grp["ground_truth"] == grp["predicted_scenario"]).all()
        n_total += 1
        n_correct += int(correct)

    lo, hi = clopper_pearson(n_correct, n_total) if n_total > 0 else (float("nan"), float("nan"))
    return {
        "rate": n_correct / n_total if n_total > 0 else float("nan"),
        "ci_lower": lo,
        "ci_upper": hi,
        "k": n_correct,
        "n": n_total,
    }


def indeterminate_sensitivity(
    records: Records,
    **filters,
) -> dict:
    """P(predicted=indeterminate | ground_truth=indeterminate_by_design).

    Measures whether the framework correctly returns indeterminate when
    interpretability is theoretically impossible — the mixed-scenario and
    transition configs labelled 'indeterminate_by_design' (spec §7).

    Parameters
    ----------
    records : list[dict] or DataFrame
        Must contain ``ground_truth`` and ``predicted_scenario``.
        Rows with ``ground_truth == 'indeterminate_by_design'`` are used.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    dict
        Keys: ``rate``, ``ci_lower``, ``ci_upper``, ``k``, ``n``.
    """
    df = _filter(_to_df(records), **filters)
    sub = df[df["ground_truth"] == "indeterminate_by_design"]
    n = len(sub)
    k = int((sub["predicted_scenario"] == "indeterminate").sum())
    lo, hi = clopper_pearson(k, n) if n > 0 else (float("nan"), float("nan"))
    return {
        "rate": k / n if n > 0 else float("nan"),
        "ci_lower": lo,
        "ci_upper": hi,
        "k": k,
        "n": n,
    }


def summary_table(
    records: Records,
    n_eval_sizes: list[int] | None = None,
    **filters,
) -> pd.DataFrame:
    """Identifiability metrics as a function of N_eval (spec §7).

    Parameters
    ----------
    records : list[dict] or DataFrame
        Evaluation records. Should contain both pure-scenario rows
        (ground_truth in SCENARIOS) and indeterminate rows
        (ground_truth == 'indeterminate_by_design') for full coverage.
    n_eval_sizes : list[int] or None
        N_eval values to report. If None, uses all unique values in records.
    **filters
        Column=value pairs passed to the internal filter.

    Returns
    -------
    pd.DataFrame
        Columns: ``n_eval``, ``strict_acc``, ``conservative_acc``,
        ``resolution_rate``, ``fscr``, ``false_confidence``,
        ``mixed_resolution_acc``, ``indet_sensitivity``, ``n_records``.
    """
    df = _filter(_to_df(records), **filters)
    if n_eval_sizes is None:
        n_eval_sizes = sorted(df["n_eval"].unique())
    rows = []
    for ne in n_eval_sizes:
        sub = df[df["n_eval"] == ne]
        sub_pure = sub[sub["ground_truth"].isin(SCENARIOS)]
        mixed_r = mixed_scenario_resolution_accuracy(sub)
        indet_r = indeterminate_sensitivity(sub)
        rows.append({
            "n_eval":               ne,
            "strict_acc":           strict_accuracy(sub_pure),
            "conservative_acc":     conservative_accuracy(sub_pure),
            "resolution_rate":      resolution_rate(sub_pure),
            "fscr":                 false_shared_claim_rate(sub_pure),
            "false_confidence":     false_confidence_rate(sub_pure),
            "mixed_resolution_acc": mixed_r["rate"],
            "indet_sensitivity":    indet_r["rate"],
            "n_records":            len(sub_pure),
        })
    return pd.DataFrame(rows)


def _filter(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Filter DataFrame by keyword column=value pairs."""
    for col, val in kwargs.items():
        if val is not None and col in df.columns:
            df = df[df[col] == val]
    return df
