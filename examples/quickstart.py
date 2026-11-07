"""DECAT quickstart on synthetic embeddings.

Builds two toy settings with paired embeddings for cohorts A', B and C and
runs the per-(model, task) DECAT evaluation on each:

- C drawn from the same distribution as B (expected verdict: Scenario 1).
- C shifted along the outcome-predictive direction, so the label boundary
  moves with cohort composition (expected verdict: Scenario 2).

The data are illustrative and contain no explicit confounder. For the
controlled synthetic setup used in the paper, see ``simulator/`` and
``experiments/synthetic/run_main_experiment.py``.

Usage::

    python examples/quickstart.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "tcga_validation"))
from run_decat import _evaluate_one


def make_cohort(
    rng: np.random.Generator, n: int, shift: float = 0.0, d: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired embeddings and binary labels for one cohort.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n : int
        Number of samples.
    shift : float, optional
        Mean shift of the shared signal; the label threshold moves with it.
    d : int, optional
        Embedding dimension per modality.

    Returns
    -------
    zh : np.ndarray
        Modality-1 embeddings, shape (n, d).
    zr : np.ndarray
        Modality-2 embeddings, shape (n, d).
    y : np.ndarray
        Binary labels, shape (n,).
    """
    s = rng.normal(size=(n, 1)) + shift
    y = (s[:, 0] + 0.5 * rng.normal(size=n) > shift).astype(int)
    zh = np.hstack([s, rng.normal(size=(n, d - 1))]) + 0.3 * rng.normal(size=(n, d))
    zr = np.hstack([s, rng.normal(size=(n, d - 1))]) + 0.3 * rng.normal(size=(n, d))
    return zh, zr, y


def main() -> None:
    for name, c_shift in [("C unshifted", 0.0), ("C shifted", 1.0)]:
        rng = np.random.default_rng(0)
        zh_ap, zr_ap, y_ap = make_cohort(rng, 300)
        zh_b, zr_b, y_b = make_cohort(rng, 300)
        zh_c, zr_c, y_c = make_cohort(rng, 300, shift=c_shift)

        records = _evaluate_one(
            split_id=0, split_type="custom", model_name="my_model",
            is_factorized=False, gamma=None,
            Zh={"Ap": zh_ap, "B": zh_b, "C": zh_c},
            Zr={"Ap": zr_ap, "B": zr_b, "C": zr_c},
            y_ap=y_ap, y_b=y_b, y_c=y_c,
            task_type="my_task", task_idx=0, seed=0,
        )
        print(name)
        for r in records:
            print(
                f"  {r['modality']}: {r['predicted_scenario']}  "
                f"A_norm={r['anorm']:.3f} (null {r['anorm_null_upper']:.3f})  "
                f"P_transfer CI_lo={r['transfer_ci_lower']:.3f} (null {r['transfer_null_upper']:.3f})  "
                f"D_q={r['dquantile_task']:.3f} (null {r['dquantile_task_null_upper']:.3f})"
            )


if __name__ == "__main__":
    main()
