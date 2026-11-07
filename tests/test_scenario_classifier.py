"""Unit tests for decision/scenario_classifier.py.

Covers every scenario (1–4), indeterminate paths, boundary conditions,
the Scenario 2 edge case, and input validation.
"""

import pytest

from decision.scenario_classifier import (
    MetricBundle,
    classify_scenario,
    SCENARIO_1,
    SCENARIO_2,
    SCENARIO_3,
    SCENARIO_4,
    INDETERMINATE,
)


# Helpers

def _factorized(**overrides) -> MetricBundle:
    """Base bundle for factorized models with all Scenario 1 conditions met.

    Derived values:
        anorm(0.5) > anorm_null_upper(0.1)               → A_norm exceeds null
        signal_present=True                               → signal gate passed
        delta_ci_lower(0.2) > delta_null_upper(0.1) > 0  → shared_dominant
        transfer_ci_lower(0.4) > transfer_null_upper(0.2) → transfer exceeds null
        dquantile_task(0.05) < dquantile_task_null_upper(0.1) → D_quantile stable
    """
    defaults = dict(
        anorm=0.5,
        anorm_null_upper=0.1,
        signal_present=True,
        transfer_ci_lower=0.4,
        transfer_null_upper=0.2,
        dquantile_task=0.05,
        dquantile_task_null_upper=0.1,
        is_factorized=True,
        delta_ci_lower=0.2,
        delta_ci_upper=0.3,
        delta_null_lower=-0.1,
        delta_null_upper=0.1,
    )
    defaults.update(overrides)
    return MetricBundle(**defaults)


def _nonfactorized(**overrides) -> MetricBundle:
    """Base bundle for non-factorized models with all Scenario 1 conditions met.

    Derived values:
        anorm(0.5) > anorm_null_upper(0.1)                → A_norm exceeds null
        signal_present=True                               → signal gate passed
        transfer_ci_lower(0.4) > transfer_null_upper(0.2) → transfer exceeds null
        dquantile_task(0.05) < dquantile_task_null_upper(0.1) → D_quantile stable
    """
    defaults = dict(
        anorm=0.5,
        anorm_null_upper=0.1,
        signal_present=True,
        transfer_ci_lower=0.4,
        transfer_null_upper=0.2,
        dquantile_task=0.05,
        dquantile_task_null_upper=0.1,
        is_factorized=False,
    )
    defaults.update(overrides)
    return MetricBundle(**defaults)


def _modality_dominant_delta():
    """Delta values that produce modality_dominant=True, shared_dominant=False.

    delta_ci_upper(-0.2) < delta_null_lower(-0.1) AND < 0 → modality_dominant
    delta_ci_lower(-0.3) > delta_null_upper(0.1)?  No     → not shared_dominant
    """
    return dict(
        delta_ci_lower=-0.3,
        delta_ci_upper=-0.2,
        delta_null_lower=-0.1,
        delta_null_upper=0.1,
    )


def _neither_dominant_delta():
    """Delta values where neither shared_dominant nor modality_dominant holds."""
    return dict(
        delta_ci_lower=-0.05,
        delta_ci_upper=0.05,
        delta_null_lower=-0.1,
        delta_null_upper=0.1,
    )


# Stage II — Signal existence gate (Scenario 3)

def test_scenario3_factorized_signal_absent():
    assert classify_scenario(_factorized(signal_present=False)) == SCENARIO_3


def test_scenario3_nonfactorized_signal_absent():
    assert classify_scenario(_nonfactorized(signal_present=False)) == SCENARIO_3


def test_scenario3_takes_priority_over_all_scenario1_conditions():
    """signal_present=False returns Scenario 3 even when all other conditions
    for Scenario 1 are satisfied."""
    m = _factorized(signal_present=False)
    assert classify_scenario(m) == SCENARIO_3


# Scenario 1

def test_scenario1_factorized_all_conditions():
    assert classify_scenario(_factorized()) == SCENARIO_1


def test_scenario1_nonfactorized_all_conditions():
    assert classify_scenario(_nonfactorized()) == SCENARIO_1


def test_scenario1_factorized_fails_without_anorm():
    m = _factorized(anorm=0.05)  # anorm(0.05) <= null_upper(0.1)
    assert classify_scenario(m) != SCENARIO_1


def test_scenario1_factorized_fails_without_shared_dominant():
    # delta_ci_lower(0.05) <= delta_null_upper(0.1) → not shared_dominant
    m = _factorized(delta_ci_lower=0.05, delta_ci_upper=0.08)
    assert classify_scenario(m) != SCENARIO_1


def test_scenario1_factorized_fails_without_transfer():
    m = _factorized(transfer_ci_lower=0.1)  # ci_lower(0.1) <= null_upper(0.2)
    assert classify_scenario(m) != SCENARIO_1


def test_scenario1_factorized_fails_without_dquantile_stable():
    m = _factorized(dquantile_task=0.15)  # dquantile(0.15) >= null_upper(0.1)
    assert classify_scenario(m) != SCENARIO_1


def test_scenario1_nonfactorized_fails_without_anorm():
    m = _nonfactorized(anorm=0.05)
    assert classify_scenario(m) != SCENARIO_1


def test_scenario1_nonfactorized_fails_without_transfer():
    m = _nonfactorized(transfer_ci_lower=0.1)
    assert classify_scenario(m) != SCENARIO_1


def test_scenario1_nonfactorized_fails_without_dquantile_stable():
    m = _nonfactorized(dquantile_task=0.15)
    assert classify_scenario(m) != SCENARIO_1


# Scenario 2

def test_scenario2_factorized():
    m = _factorized(dquantile_task=0.15)  # NOT stable → Scenario 2
    assert classify_scenario(m) == SCENARIO_2


def test_scenario2_nonfactorized():
    m = _nonfactorized(dquantile_task=0.15)
    assert classify_scenario(m) == SCENARIO_2


def test_scenario2_nonfactorized_even_with_anorm():
    """A_norm present but D_quantile unstable → Scenario 2, not 1."""
    m = _nonfactorized(dquantile_task=0.15)
    assert classify_scenario(m) == SCENARIO_2


def test_scenario2_factorized_localization_not_required():
    """Scenario 2 factorized: neither shared nor modality dominant — still valid.


    """
    m = _factorized(**_neither_dominant_delta(), dquantile_task=0.15)
    assert classify_scenario(m) == SCENARIO_2


def test_scenario2_shared_dominant_no_anorm_unstable():
    """Key edge case: shared_dominant=True, A_norm fails, D_quantile unstable.

    The shared-dominant localization is rejected as a factorization artifact
    (A_norm does not exceed its null), but the signal character — transferable
    with unstable biological ordering — is still correctly captured as Scenario 2.

    Must NOT return INDETERMINATE.
    """
    m = _factorized(
        anorm=0.05,    # does NOT exceed null_upper(0.1)
        dquantile_task=0.15,  # NOT stable → D_quantile >= null_upper
    )
    assert classify_scenario(m) == SCENARIO_2


# Scenario 4

def test_scenario4_factorized():
    m = _factorized(**_modality_dominant_delta())
    assert classify_scenario(m) == SCENARIO_4


def test_scenario4_nonfactorized():
    """Non-factorized Scenario 4: transfer + stability but A_norm below null."""
    m = _nonfactorized(anorm=0.05)  # anorm(0.05) <= null_upper(0.1) → not Scenario 1
    assert classify_scenario(m) == SCENARIO_4


def test_scenario4_factorized_requires_modality_dominant():
    """Factorized: neither dominant, transfer + stable → Indeterminate (not 4)."""
    m = _factorized(**_neither_dominant_delta(), anorm=0.05)
    assert classify_scenario(m) == INDETERMINATE


def test_scenario4_factorized_requires_transfer():
    m = _factorized(**_modality_dominant_delta(), transfer_ci_lower=0.1)
    assert classify_scenario(m) != SCENARIO_4


def test_scenario4_factorized_requires_dquantile_stable():
    m = _factorized(**_modality_dominant_delta(), dquantile_task=0.15)
    assert classify_scenario(m) != SCENARIO_4


def test_scenario4_nonfactorized_upgrades_to_scenario1_when_anorm_present():
    """Non-factorized: if A_norm present + transfer + stable → Scenario 1, not 4."""
    assert classify_scenario(_nonfactorized()) == SCENARIO_1


# Indeterminate

def test_indeterminate_factorized_transfer_fails():
    m = _factorized(transfer_ci_lower=0.1)
    assert classify_scenario(m) == INDETERMINATE


def test_indeterminate_nonfactorized_transfer_fails():
    m = _nonfactorized(transfer_ci_lower=0.1)
    assert classify_scenario(m) == INDETERMINATE


def test_indeterminate_shared_dominant_no_anorm_stable():
    """shared_dominant=True, A_norm fails, D_quantile stable.

    Scenario 1: fails — no A_norm.
    Scenario 2: fails — D_quantile is stable (within null).
    Scenario 4: fails — modality_dominant=False (shared_dominant=True).
    → Indeterminate.
    """
    m = _factorized(anorm=0.05)  # shared_dominant=True from base; dquantile stable
    assert classify_scenario(m) == INDETERMINATE


def test_indeterminate_factorized_neither_dominant_stable():
    """Factorized: neither localization succeeds, D_quantile stable → Indeterminate."""
    m = _factorized(**_neither_dominant_delta(), anorm=0.05)
    assert classify_scenario(m) == INDETERMINATE


def test_indeterminate_nonfactorized_no_anorm_no_transfer():
    m = _nonfactorized(anorm=0.05, transfer_ci_lower=0.1)
    assert classify_scenario(m) == INDETERMINATE


# Boundary conditions (strict inequality checks)

def test_boundary_anorm_exactly_at_null_upper_is_not_exceeded():
    """anorm == anorm_null_upper uses strict >, so it does NOT exceed null."""
    m = _nonfactorized(anorm=0.1, anorm_null_upper=0.1)
    assert classify_scenario(m) != SCENARIO_1


def test_boundary_dquantile_exactly_at_null_upper_is_not_stable():
    """dquantile == dquantile_null_upper uses strict <, so it is NOT stable → Scenario 2."""
    m = _nonfactorized(dquantile_task=0.1, dquantile_task_null_upper=0.1)
    assert classify_scenario(m) == SCENARIO_2


def test_boundary_transfer_ci_lower_exactly_at_null_upper_does_not_exceed():
    """transfer_ci_lower == transfer_null_upper uses strict >, so does NOT exceed."""
    m = _nonfactorized(transfer_ci_lower=0.2, transfer_null_upper=0.2)
    assert classify_scenario(m) == INDETERMINATE


def test_boundary_delta_ci_lower_exactly_at_null_upper_is_not_shared_dominant():
    """delta_ci_lower == delta_null_upper uses strict >, so not shared_dominant."""
    m = _factorized(delta_ci_lower=0.1, delta_null_upper=0.1)
    assert classify_scenario(m) != SCENARIO_1


# Input validation

def test_valueerror_is_factorized_all_delta_fields_none():
    with pytest.raises(ValueError, match="delta_ci_lower"):
        classify_scenario(MetricBundle(
            anorm=0.5,
            anorm_null_upper=0.1,
            signal_present=True,
            transfer_ci_lower=0.4,
            transfer_null_upper=0.2,
            dquantile_task=0.05,
            dquantile_task_null_upper=0.1,
            is_factorized=True,
            # all delta fields default to None
        ))


def test_valueerror_is_factorized_partial_delta_fields():
    """Providing only some delta fields still raises ValueError."""
    with pytest.raises(ValueError):
        classify_scenario(MetricBundle(
            anorm=0.5,
            anorm_null_upper=0.1,
            signal_present=True,
            transfer_ci_lower=0.4,
            transfer_null_upper=0.2,
            dquantile_task=0.05,
            dquantile_task_null_upper=0.1,
            is_factorized=True,
            delta_ci_lower=0.2,
            # delta_ci_upper, delta_null_lower, delta_null_upper left None
        ))


def test_no_valueerror_nonfactorized_delta_fields_none():
    """is_factorized=False with no delta fields is valid."""
    m = _nonfactorized()
    assert classify_scenario(m) in {SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4, INDETERMINATE}
