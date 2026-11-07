"""Decision tree for DECAT scenario classification.

Implements the rule-based DECAT decision procedure,
mapping pre-computed interpretability metrics to biological scenario
labels.

Applied per task, per modality.
"""

from dataclasses import dataclass
from typing import Optional


SCENARIO_1 = "scenario_1"
SCENARIO_2 = "scenario_2"
SCENARIO_3 = "scenario_3"
SCENARIO_4 = "scenario_4"
INDETERMINATE = "indeterminate"


@dataclass
class MetricBundle:
    """Pre-computed DECAT metrics for one (model, task, modality) combination.

    Attributes
    ----------
    anorm : float
        Observed A_norm (cross-modal shared agreement).
    anorm_null_upper : float
        97.5th percentile of the A_norm permutation null.
    signal_present : bool
        Whether the label-permutation signal-presence gate passed (p < 0.05).
    transfer_ci_lower : float
        Lower bootstrap CI bound for P_transfer.
    transfer_null_upper : float
        97.5th percentile of the P_transfer permutation null.
    dquantile_task : float
        Observed D_quantile^task (probe direction, A'-referenced; Metric D3).
    dquantile_task_null_upper : float
        97.5th percentile of the D_quantile^task permutation null.
    is_factorized : bool
        Whether the model provides an explicit shared/modality-specific split.
    delta_ci_lower : float or None
        Lower bootstrap CI bound for Delta_shared.  Required if is_factorized.
    delta_ci_upper : float or None
        Upper bootstrap CI bound for Delta_shared.  Required if is_factorized.
    delta_null_lower : float or None
        2.5th percentile of the Delta_shared permutation null.
        Required if is_factorized.
    delta_null_upper : float or None
        97.5th percentile of the Delta_shared permutation null.
        Required if is_factorized.
    """

    anorm: float
    anorm_null_upper: float
    signal_present: bool
    transfer_ci_lower: float
    transfer_null_upper: float
    dquantile_task: float
    dquantile_task_null_upper: float
    is_factorized: bool
    delta_ci_lower: Optional[float] = None
    delta_ci_upper: Optional[float] = None
    delta_null_lower: Optional[float] = None
    delta_null_upper: Optional[float] = None


def classify_scenario(metrics: MetricBundle) -> str:
    """Apply the DECAT decision tree to a bundle of pre-computed metrics.

    Applied per task, per modality.

    Parameters
    ----------
    metrics : MetricBundle

    Returns
    -------
    str
        One of "scenario_1", "scenario_2", "scenario_3", "scenario_4",
        or "indeterminate".

    Raises
    ------
    ValueError
        If is_factorized=True but any required Delta_shared field is None.
    """
    if metrics.is_factorized:
        _validate_factorized_fields(metrics)

    # Pre-compute Stage IV metric comparisons unconditionally.
    # D_quantile^task is now computed unconditionally in _compute_stage4
    # (see rationale there), so S2 can be detected even when signal_present
    # is False — the S2 check must therefore precede the signal_present→S3
    # gate rather than being gated behind it.
    anorm_exceeds_null    = metrics.anorm > metrics.anorm_null_upper
    transfer_exceeds_null = metrics.transfer_ci_lower > metrics.transfer_null_upper
    dquantile_within_null = metrics.dquantile_task < metrics.dquantile_task_null_upper

    # Stage III — Predictive localization (factorized models only).
    if metrics.is_factorized:
        shared_dominant   = is_shared_dominant(metrics.delta_ci_lower, metrics.delta_null_upper)
        modality_dominant = is_modality_dominant(metrics.delta_ci_upper, metrics.delta_null_lower)

    # Scenario 2 — checked BEFORE the signal_present→S3 gate.
    # Prediction transfers but biological ordering does not.  Must precede
    # Stage II because cohort-level shift (dq_task) can fire even when
    # individual-level AUROC is marginal (signal_present=False).
    if transfer_exceeds_null and not dquantile_within_null:
        return SCENARIO_2

    # Stage II — Signal existence gate.
    # Reached only when S2 conditions not met.  No predictive signal → S3.
    if not metrics.signal_present:
        return SCENARIO_3

    # Stage IV — Cross-cohort stability + scenario assignment (S1 and S4).

    # Scenario 1 — shared biology, transferable, biologically ordered
    if metrics.is_factorized:
        if (
            anorm_exceeds_null
            and shared_dominant
            and transfer_exceeds_null
            and dquantile_within_null
        ):
            return SCENARIO_1
    else:
        if anorm_exceeds_null and transfer_exceeds_null and dquantile_within_null:
            return SCENARIO_1

    # Scenario 4 — modality-specific, transferable, biologically ordered
    if metrics.is_factorized:
        if modality_dominant and transfer_exceeds_null and dquantile_within_null:
            return SCENARIO_4
    else:
        # For non-factorized models, "NOT Scenario 1" is guaranteed by control
        # flow: if Scenario 1 conditions were met we would have returned above.
        if transfer_exceeds_null and dquantile_within_null:
            return SCENARIO_4

    return INDETERMINATE


def is_shared_dominant(ci_lower: Optional[float], null_upper: Optional[float]) -> bool:
    """Return True if Delta_shared CI lower bound exceeds the permutation null upper bound."""
    return (ci_lower is not None and null_upper is not None
            and ci_lower > null_upper and ci_lower > 0.0)


def is_modality_dominant(ci_upper: Optional[float], null_lower: Optional[float]) -> bool:
    """Return True if Delta_shared CI upper bound falls below the permutation null lower bound."""
    return (ci_upper is not None and null_lower is not None
            and ci_upper < null_lower and ci_upper < 0.0)


def _validate_factorized_fields(metrics: MetricBundle) -> None:
    """Raise ValueError if any Delta_shared field required for factorized models is None."""
    missing = [
        name
        for name in (
            "delta_ci_lower",
            "delta_ci_upper",
            "delta_null_lower",
            "delta_null_upper",
        )
        if getattr(metrics, name) is None
    ]
    if missing:
        raise ValueError(
            f"is_factorized=True but the following required fields are None: {missing}"
        )
