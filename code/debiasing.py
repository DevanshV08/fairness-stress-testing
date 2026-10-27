"""Debiasing methods: Reweighing, ExponentiatedGradient, ThresholdOptimizer."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer


class ProbabilityDtypeAdapter(BaseEstimator):
    """Wrap an estimator and normalize predict_proba output to float64 ndarray."""

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> "ProbabilityDtypeAdapter":
        self.estimator.fit(X, y, **kwargs)
        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X), dtype=np.float64)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"estimator": self.estimator}

    def set_params(self, **params: Any) -> "ProbabilityDtypeAdapter":
        if "estimator" in params:
            self.estimator = params["estimator"]
        return self


# ---------------------------------------------------------------------------
# Reweighing  (Kamiran & Calders 2012)
# ---------------------------------------------------------------------------

def compute_reweighing_weights(
    y: np.ndarray,
    sensitive: np.ndarray,
) -> np.ndarray:
    """Compute per-sample weights that equalise group × label joint probabilities.

    Parameters
    ----------
    y : array of shape (n,)
        Binary labels.
    sensitive : array of shape (n,)
        Binary sensitive attribute.

    Returns
    -------
    weights : array of shape (n,)
        Sample weights in [0, ∞).
    """
    y = np.asarray(y)
    s = np.asarray(sensitive)
    n = len(y)

    weights = np.ones(n, dtype=float)
    for s_val in np.unique(s):
        for y_val in np.unique(y):
            mask = (s == s_val) & (y == y_val)
            p_s = (s == s_val).mean()
            p_y = (y == y_val).mean()
            p_sy = mask.mean()
            if p_sy > 0:
                weights[mask] = (p_s * p_y) / p_sy
    return weights


def apply_reweighing(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
) -> BaseEstimator:
    """Fit ``estimator`` with reweighing sample weights.

    Parameters
    ----------
    estimator : unfitted sklearn-compatible estimator
    X_train, y_train, sensitive_train : training data

    Returns
    -------
    Fitted estimator.
    """
    weights = compute_reweighing_weights(y_train, sensitive_train)
    est = clone(estimator)
    try:
        est.fit(X_train, y_train, sample_weight=weights)
    except TypeError:
        # FT-Transformer wrapper accepts sample_weight differently
        est.fit(X_train, y_train, sample_weight=weights)
    return est


# ---------------------------------------------------------------------------
# ExponentiatedGradient  (Agarwal et al. 2018)
# ---------------------------------------------------------------------------

def apply_exp_gradient(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    constraint: str = "demographic_parity",
    eps: float = 0.01,
) -> ExponentiatedGradient:
    """Fit a fairlearn ExponentiatedGradient mitigator.

    Parameters
    ----------
    estimator : unfitted sklearn estimator (base learner)
    X_train, y_train, sensitive_train : training data
    constraint : ``'demographic_parity'`` or ``'equalized_odds'``
    eps : fairness violation tolerance

    Returns
    -------
    Fitted ExponentiatedGradient (sklearn-compatible).
    """
    if constraint == "demographic_parity":
        c = DemographicParity(difference_bound=eps)
    elif constraint == "equalized_odds":
        c = EqualizedOdds(difference_bound=eps)
    else:
        raise ValueError(f"Unknown constraint: {constraint!r}")

    mitigator = ExponentiatedGradient(clone(estimator), constraints=c)
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    return mitigator


# ---------------------------------------------------------------------------
# ThresholdOptimizer  (Hardt et al. 2016)
# ---------------------------------------------------------------------------

def apply_threshold_optimizer(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    constraint: str = "demographic_parity",
) -> ThresholdOptimizer:
    """Fit a fairlearn ThresholdOptimizer post-processor.

    The base estimator is first fitted without fairness constraints, then
    post-processing thresholds are derived on the training set.

    Parameters
    ----------
    estimator : unfitted sklearn estimator
    X_train, y_train, sensitive_train : training data
    constraint : ``'demographic_parity'`` or ``'equalized_odds'``

    Returns
    -------
    Fitted ThresholdOptimizer.
    """
    base = ProbabilityDtypeAdapter(clone(estimator))
    base.fit(X_train, y_train)

    postproc = ThresholdOptimizer(
        estimator=base,
        constraints=constraint,
        predict_method="predict_proba",
        objective="accuracy_score",
    )
    postproc.fit(X_train, y_train, sensitive_features=sensitive_train)
    return postproc


# ---------------------------------------------------------------------------
# Dispatch helper used by pipeline.py
# ---------------------------------------------------------------------------

DEBIAS_METHODS = ("none", "reweighing", "exp_gradient", "threshold_optimizer")


def build_debiased_model(
    method: str,
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    constraint: str = "demographic_parity",
) -> BaseEstimator:
    """Fit and return a (potentially debiased) model.

    Parameters
    ----------
    method : one of ``DEBIAS_METHODS``
    estimator : unfitted base estimator from ``models.build_model``
    X_train, y_train, sensitive_train : training split
    constraint : fairness constraint for ExpGradient / ThresholdOptimizer

    Returns
    -------
    Fitted estimator.
    """
    if method == "none":
        est = clone(estimator)
        est.fit(X_train, y_train)
        return est
    if method == "reweighing":
        return apply_reweighing(estimator, X_train, y_train, sensitive_train)
    if method == "exp_gradient":
        return apply_exp_gradient(estimator, X_train, y_train, sensitive_train, constraint)
    if method == "threshold_optimizer":
        return apply_threshold_optimizer(estimator, X_train, y_train, sensitive_train, constraint)
    raise ValueError(f"Unknown debias method: {method!r}. Choose from {DEBIAS_METHODS}.")
