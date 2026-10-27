"""Fairness metrics: DP diff, EO diff, equalized odds diff, per-group stats, ECE."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Difference in selection rates between privileged and unprivileged groups.

    Parameters
    ----------
    y_pred : array of shape (n,)
        Binary predictions (0/1).
    sensitive : array of shape (n,)
        Binary sensitive attribute (1 = privileged).

    Returns
    -------
    float
        Selection rate (privileged) − selection rate (unprivileged).
    """
    s = np.asarray(sensitive)
    p = np.asarray(y_pred)
    priv = p[s == 1].mean()
    unpriv = p[s == 0].mean()
    return float(priv - unpriv)


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Difference in TPR (recall) between privileged and unprivileged groups.

    Parameters
    ----------
    y_true : array of shape (n,)
    y_pred : array of shape (n,)
    sensitive : array of shape (n,)

    Returns
    -------
    float
        TPR (privileged) − TPR (unprivileged).
    """
    s, y, p = np.asarray(sensitive), np.asarray(y_true), np.asarray(y_pred)

    def tpr(mask: np.ndarray) -> float:
        pos = mask & (y == 1)
        return float(p[pos].mean()) if pos.any() else np.nan

    return float(tpr(s == 1) - tpr(s == 0))


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Max of |TPR diff| and |FPR diff| between privileged and unprivileged groups.

    Returns
    -------
    float
        max(|TPR_priv − TPR_unpriv|, |FPR_priv − FPR_unpriv|).
    """
    s, y, p = np.asarray(sensitive), np.asarray(y_true), np.asarray(y_pred)

    def tpr(mask: np.ndarray) -> float:
        pos = mask & (y == 1)
        return float(p[pos].mean()) if pos.any() else np.nan

    def fpr(mask: np.ndarray) -> float:
        neg = mask & (y == 0)
        return float(p[neg].mean()) if neg.any() else np.nan

    tpr_diff = abs(tpr(s == 1) - tpr(s == 0))
    fpr_diff = abs(fpr(s == 1) - fpr(s == 0))
    return float(np.nanmax([tpr_diff, fpr_diff]))


def per_group_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Per-group accuracy, selection rate, TPR, FPR (and ECE if probabilities given).

    Returns
    -------
    pd.DataFrame with one row per unique group value.
    """
    s = np.asarray(sensitive)
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    rows = []
    for g in sorted(np.unique(s)):
        mask = s == g
        gp, gy = p[mask], y[mask]
        pos_mask = gy == 1
        neg_mask = gy == 0
        row: Dict = {
            "group": int(g),
            "n": int(mask.sum()),
            "selection_rate": float(gp.mean()),
            "accuracy": float((gp == gy).mean()),
            "tpr": float(gp[pos_mask].mean()) if pos_mask.any() else np.nan,
            "fpr": float(gp[neg_mask].mean()) if neg_mask.any() else np.nan,
        }
        if y_prob is not None:
            row["ece"] = _ece(gy, y_prob[mask])
        rows.append(row)
    return pd.DataFrame(rows)


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    weights = bin_counts / bin_counts.sum()
    return float(np.sum(weights[: len(prob_true)] * np.abs(prob_true - prob_pred)))


def calibration_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Public wrapper for Expected Calibration Error."""
    return _ece(np.asarray(y_true), np.asarray(y_prob), n_bins=n_bins)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return all scalar fairness + utility metrics in one dict.

    Keys: accuracy, [auc], dp_diff, eo_diff, eod_diff, [ece].
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "dp_diff": demographic_parity_difference(y_pred, sensitive),
        "eo_diff": equal_opportunity_difference(y_true, y_pred, sensitive),
        "eod_diff": equalized_odds_difference(y_true, y_pred, sensitive),
    }
    if y_prob is not None:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["ece"] = calibration_ece(y_true, y_prob)
    return out
