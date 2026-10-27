"""Stress-testing layer: seed sweep, threshold sweep, intersectional breakdown."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loaders import load_adult, load_german, load_compas
from models import build_model
from debiasing import build_debiased_model
from fairness_metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
    compute_all_metrics,
)

from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOADERS = {"adult": load_adult, "german": load_german, "compas": load_compas}
PRIMARY_SENSITIVE = {"adult": "sex", "german": "sex", "compas": "race"}

# Intersectional pairs: (attr1, attr2)
INTERSECTIONAL_PAIRS = {
    "adult": ("sex", "race"),
    "compas": ("race", "sex"),   # sex is a feature but also useful here
    "german": ("sex", "age"),
}

MIN_GROUP_SIZE_WARN = 100
MIN_GROUP_SIZE_DROP = 30


def _predict_with_optional_sensitive(
    estimator: Any,
    X_test: pd.DataFrame,
    sensitive_test: np.ndarray,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Predict labels and probabilities, handling Fairlearn post-processors."""
    predict_kwargs: Dict[str, Any] = {}
    if estimator.__class__.__name__ == "ThresholdOptimizer":
        predict_kwargs["sensitive_features"] = sensitive_test

    y_pred = estimator.predict(X_test, **predict_kwargs)

    y_prob: Optional[np.ndarray] = None
    if hasattr(estimator, "predict_proba"):
        try:
            y_prob = estimator.predict_proba(X_test, **predict_kwargs)[:, 1]
        except Exception:
            pass

    return y_pred, y_prob


def _load_and_split(
    dataset: str,
    seed: int,
    test_size: float = 0.2,
    scale: bool = True,
):
    """Shared helper: load, split, scale. Returns train/test splits."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    X, y, sensitive_attrs = LOADERS[dataset]()
    X_arr = X.values.astype(np.float64)
    y_arr = y.values

    splits = train_test_split(X_arr, y_arr, sensitive_attrs.values,
                              test_size=test_size, random_state=seed, stratify=y_arr)
    X_tr, X_te, y_tr, y_te, s_tr, s_te = splits

    if scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    return X, y, sensitive_attrs, X_tr, X_te, y_tr, y_te, s_tr, s_te


# ---------------------------------------------------------------------------
# 1. Seed sweep
# ---------------------------------------------------------------------------

def run_seed_sweep(
    dataset: str,
    model_name: str,
    debias_method: str = "none",
    seeds: List[int] = None,
    sensitive_col: Optional[str] = None,
    save: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run one (dataset × model × debias) config across multiple seeds.

    Returns
    -------
    pd.DataFrame with one row per seed; columns include all scalar metrics.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    col = sensitive_col or PRIMARY_SENSITIVE[dataset]
    rows = []

    for seed in seeds:
        X, y, sensitive_attrs, X_tr, X_te, y_tr, y_te, s_tr, s_te = _load_and_split(
            dataset, seed
        )
        s_tr_col = sensitive_attrs.iloc[: len(y_tr)][col].values  # approximate; reindex below

        # Proper per-seed reindex via split indices
        full_s = sensitive_attrs[col].values
        X_arr = X.values.astype(np.float64)
        y_arr = y.values
        from sklearn.model_selection import train_test_split as _split
        import random as _r
        _r.seed(seed)
        np.random.seed(seed)
        idx = np.arange(len(y_arr))
        idx_tr, idx_te = _split(idx, test_size=0.2, random_state=seed, stratify=y_arr)
        sc = StandardScaler()
        X_tr2 = sc.fit_transform(X_arr[idx_tr])
        X_te2 = sc.transform(X_arr[idx_te])
        s_tr2, s_te2 = full_s[idx_tr], full_s[idx_te]
        y_tr2, y_te2 = y_arr[idx_tr], y_arr[idx_te]

        X_tr_df = pd.DataFrame(X_tr2, columns=X.columns, dtype=np.float64)
        X_te_df = pd.DataFrame(X_te2, columns=X.columns, dtype=np.float64)

        estimator = build_model(model_name, seed=seed)
        fitted = build_debiased_model(debias_method, estimator, X_tr_df, y_tr2, s_tr2)

        y_pred, y_prob = _predict_with_optional_sensitive(fitted, X_te_df, s_te2)

        metrics = compute_all_metrics(y_te2, y_pred, s_te2, y_prob)
        rows.append({"seed": seed, "dataset": dataset, "model": model_name,
                     "debias_method": debias_method, "sensitive_col": col, **metrics})

    df = pd.DataFrame(rows)
    if save:
        out = RESULTS_DIR / f"seed_sweep_{dataset}_{model_name}_{debias_method}_{col}.csv"
        df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------
# 2. Threshold sweep
# ---------------------------------------------------------------------------

def run_threshold_sweep(
    dataset: str,
    model_name: str,
    debias_method: str = "none",
    seed: int = 0,
    sensitive_col: Optional[str] = None,
    thresholds: Optional[np.ndarray] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Sweep decision threshold ∈ [0.3, 0.7] and record fairness-accuracy curves.

    Only meaningful for probabilistic classifiers (LR, XGBoost, FT-Transformer).

    Returns
    -------
    pd.DataFrame with one row per threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.05)

    col = sensitive_col or PRIMARY_SENSITIVE[dataset]
    X, y, sensitive_attrs = LOADERS[dataset]()
    full_s = sensitive_attrs[col].values
    X_arr = X.values.astype(np.float64)
    y_arr = y.values

    idx = np.arange(len(y_arr))
    from sklearn.model_selection import train_test_split as _split
    idx_tr, idx_te = _split(idx, test_size=0.2, random_state=seed, stratify=y_arr)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_arr[idx_tr])
    X_te = sc.transform(X_arr[idx_te])
    y_tr, y_te = y_arr[idx_tr], y_arr[idx_te]
    s_tr, s_te = full_s[idx_tr], full_s[idx_te]

    X_tr_df = pd.DataFrame(X_tr, columns=X.columns, dtype=np.float64)
    X_te_df = pd.DataFrame(X_te, columns=X.columns, dtype=np.float64)

    estimator = build_model(model_name, seed=seed)
    fitted = build_debiased_model(debias_method, estimator, X_tr_df, y_tr, s_tr)

    if fitted.__class__.__name__ == "ThresholdOptimizer":
        return pd.DataFrame()

    if not hasattr(fitted, "predict_proba"):
        warnings.warn(f"{model_name} has no predict_proba; threshold sweep not meaningful.")
        return pd.DataFrame()

    try:
        y_prob = fitted.predict_proba(X_te_df)[:, 1]
    except Exception:
        return pd.DataFrame()

    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        from sklearn.metrics import accuracy_score
        rows.append({
            "threshold": round(float(t), 2),
            "accuracy": float(accuracy_score(y_te, y_pred_t)),
            "dp_diff": demographic_parity_difference(y_pred_t, s_te),
            "eo_diff": equal_opportunity_difference(y_te, y_pred_t, s_te),
            "eod_diff": equalized_odds_difference(y_te, y_pred_t, s_te),
        })

    df = pd.DataFrame(rows)
    if save:
        out = RESULTS_DIR / f"threshold_sweep_{dataset}_{model_name}_{debias_method}_{col}_seed{seed}.csv"
        df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. Intersectional breakdown
# ---------------------------------------------------------------------------

def run_intersectional(
    dataset: str,
    model_name: str,
    debias_method: str = "none",
    seed: int = 0,
    save: bool = True,
) -> pd.DataFrame:
    """Report per-subgroup (attr1 × attr2) metrics.

    Subgroups with n < MIN_GROUP_SIZE_DROP are dropped; n < MIN_GROUP_SIZE_WARN are flagged.

    Returns
    -------
    pd.DataFrame with one row per (group_a, group_b) subgroup.
    """
    attr1, attr2 = INTERSECTIONAL_PAIRS[dataset]
    X, y, sensitive_attrs = LOADERS[dataset]()

    # For COMPAS, sex is a feature but not in sensitive_attrs — derive it
    if dataset == "compas" and attr2 == "sex":
        # sex was encoded as int in X already
        if "sex" in X.columns:
            sensitive_attrs = sensitive_attrs.copy()
            sensitive_attrs["sex"] = X["sex"].values
        else:
            sensitive_attrs = sensitive_attrs.copy()
            sensitive_attrs["sex"] = np.nan

    full_s1 = sensitive_attrs[attr1].values
    full_s2 = sensitive_attrs[attr2].values
    X_arr = X.values.astype(np.float64)
    y_arr = y.values

    idx = np.arange(len(y_arr))
    from sklearn.model_selection import train_test_split as _split
    idx_tr, idx_te = _split(idx, test_size=0.2, random_state=seed, stratify=y_arr)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_arr[idx_tr])
    X_te = sc.transform(X_arr[idx_te])
    y_tr, y_te = y_arr[idx_tr], y_arr[idx_te]
    s1_tr = full_s1[idx_tr]
    s1_te, s2_te = full_s1[idx_te], full_s2[idx_te]

    X_tr_df = pd.DataFrame(X_tr, columns=X.columns, dtype=np.float64)
    X_te_df = pd.DataFrame(X_te, columns=X.columns, dtype=np.float64)

    estimator = build_model(model_name, seed=seed)
    fitted = build_debiased_model(debias_method, estimator, X_tr_df, y_tr, s1_tr)

    y_pred, y_prob = _predict_with_optional_sensitive(fitted, X_te_df, s1_te)

    rows = []
    for v1 in sorted(np.unique(s1_te)):
        for v2 in sorted(np.unique(s2_te)):
            mask = (s1_te == v1) & (s2_te == v2)
            n = mask.sum()
            if n < MIN_GROUP_SIZE_DROP:
                continue
            flag = n < MIN_GROUP_SIZE_WARN
            gp, gy = y_pred[mask], y_te[mask]
            from sklearn.metrics import accuracy_score
            row: Dict[str, Any] = {
                "dataset": dataset,
                "model": model_name,
                "debias_method": debias_method,
                "seed": seed,
                f"{attr1}": int(v1),
                f"{attr2}": int(v2),
                "n": int(n),
                "small_group_warning": flag,
                "selection_rate": float(gp.mean()),
                "accuracy": float(accuracy_score(gy, gp)),
                "tpr": float(gp[gy == 1].mean()) if (gy == 1).any() else np.nan,
                "fpr": float(gp[gy == 0].mean()) if (gy == 0).any() else np.nan,
            }
            if y_prob is not None:
                row["auc"] = _safe_auc(gy, y_prob[mask])
            rows.append(row)

    df = pd.DataFrame(rows)
    if save:
        out = RESULTS_DIR / f"intersectional_{dataset}_{model_name}_{debias_method}_seed{seed}.csv"
        df.to_csv(out, index=False)
    return df


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return np.nan
