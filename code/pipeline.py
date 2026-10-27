"""Single-seed end-to-end pipeline: load → split → model → debias → evaluate."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loaders import load_adult, load_german, load_compas
from models import build_model
from debiasing import build_debiased_model, DEBIAS_METHODS
from fairness_metrics import compute_all_metrics, per_group_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOADERS = {
    "adult": load_adult,
    "german": load_german,
    "compas": load_compas,
}

# Primary sensitive attribute per dataset (used when caller doesn't specify)
PRIMARY_SENSITIVE = {
    "adult": "sex",
    "german": "sex",
    "compas": "race",
}


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


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


def run_single(
    dataset: str,
    model_name: str,
    debias_method: str = "none",
    seed: int = 0,
    sensitive_col: Optional[str] = None,
    test_size: float = 0.2,
    constraint: str = "demographic_parity",
    scale_features: bool = True,
    save: bool = True,
) -> Dict[str, Any]:
    """Run one (dataset × model × debias × seed) experiment.

    Parameters
    ----------
    dataset : ``'adult'``, ``'german'``, or ``'compas'``
    model_name : ``'lr'``, ``'rf'``, ``'xgboost'``, or ``'ft_transformer'``
    debias_method : one of ``debiasing.DEBIAS_METHODS``
    seed : random seed
    sensitive_col : column in sensitive_attrs to use; defaults to PRIMARY_SENSITIVE
    test_size : fraction held out for evaluation
    constraint : fairness constraint passed to ExpGradient / ThresholdOptimizer
    scale_features : apply StandardScaler (recommended for LR / FT-Transformer)
    save : write result JSON to results/

    Returns
    -------
    dict with keys: dataset, model, debias_method, seed, sensitive_col,
                    accuracy, [auc], dp_diff, eo_diff, eod_diff, [ece],
                    + per-group stats as nested dicts.
    """
    assert dataset in LOADERS, f"Unknown dataset {dataset!r}"
    assert debias_method in DEBIAS_METHODS, f"Unknown method {debias_method!r}"

    _set_global_seeds(seed)

    # ---- Load ---------------------------------------------------------------
    X, y, sensitive_attrs = LOADERS[dataset]()
    col = sensitive_col or PRIMARY_SENSITIVE[dataset]
    sensitive = sensitive_attrs[col].values

    # ---- Split (stratified on label) ----------------------------------------
    X_arr = X.values.astype(np.float64)
    y_arr = y.values

    (X_train, X_test,
     y_train, y_test,
     s_train, s_test) = train_test_split(
        X_arr, y_arr, sensitive,
        test_size=test_size,
        random_state=seed,
        stratify=y_arr,
    )

    # ---- Scale --------------------------------------------------------------
    scaler: Optional[StandardScaler] = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train, columns=X.columns, dtype=np.float64)
    X_test_df = pd.DataFrame(X_test, columns=X.columns, dtype=np.float64)

    # ---- Model --------------------------------------------------------------
    estimator = build_model(model_name, seed=seed)

    fitted = build_debiased_model(
        debias_method,
        estimator,
        X_train_df,
        y_train,
        s_train,
        constraint=constraint,
    )

    # ---- Predict ------------------------------------------------------------
    y_pred, y_prob = _predict_with_optional_sensitive(fitted, X_test_df, s_test)

    # ---- Metrics ------------------------------------------------------------
    metrics = compute_all_metrics(y_test, y_pred, s_test, y_prob)
    groups_df = per_group_stats(y_test, y_pred, s_test, y_prob)

    result: Dict[str, Any] = {
        "dataset": dataset,
        "model": model_name,
        "debias_method": debias_method,
        "seed": seed,
        "sensitive_col": col,
        **{k: round(v, 4) for k, v in metrics.items()},
        "per_group": groups_df.to_dict(orient="records"),
    }

    if save:
        fname = (
            RESULTS_DIR
            / f"{dataset}_{model_name}_{debias_method}_{col}_seed{seed}.json"
        )
        fname.write_text(json.dumps(result, indent=2))

    return result


def run_grid(
    datasets: list[str],
    models: list[str],
    debias_methods: list[str],
    seeds: list[int],
    **kwargs: Any,
) -> pd.DataFrame:
    """Run all combinations and return a tidy DataFrame.

    Parameters
    ----------
    datasets, models, debias_methods, seeds : lists of values to cross
    **kwargs : forwarded to ``run_single``

    Returns
    -------
    pd.DataFrame with one row per (dataset x model x method x seed).
    """
    rows = []
    total = len(datasets) * len(models) * len(debias_methods) * len(seeds)
    done = 0
    for ds in datasets:
        for mdl in models:
            for mth in debias_methods:
                for sd in seeds:
                    print(f"[{done+1}/{total}] {ds} | {mdl} | {mth} | seed={sd}")
                    try:
                        row = run_single(ds, mdl, mth, sd, **kwargs)
                        rows.append(row)
                    except Exception as exc:
                        print(f"  ERROR: {exc}")
                        rows.append({
                            "dataset": ds, "model": mdl,
                            "debias_method": mth, "seed": sd, "error": str(exc),
                        })
                    done += 1

    df = pd.json_normalize(rows)
    out = RESULTS_DIR / "grid_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows -> {out}")
    return df
