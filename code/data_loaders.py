"""Dataset loaders for Adult Income, German Credit, and COMPAS."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"Downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)


# ---------------------------------------------------------------------------
# Adult Income
# ---------------------------------------------------------------------------

_ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

_ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
_ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


def load_adult(
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load and preprocess the Adult Income dataset.

    Parameters
    ----------
    data_dir : Path
        Directory to cache raw files.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (sensitive attributes excluded).
    y : pd.Series
        Binary target: 1 = income >50K (favorable).
    sensitive_attrs : pd.DataFrame
        Columns: ``sex`` (Male=1/Female=0), ``race`` (White=1/non-White=0).
    """
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    _download(_ADULT_TRAIN_URL, train_path)
    _download(_ADULT_TEST_URL, test_path)

    train = pd.read_csv(train_path, names=_ADULT_COLUMNS, na_values=" ?", skipinitialspace=True)
    test = pd.read_csv(test_path, names=_ADULT_COLUMNS, na_values=" ?", skipinitialspace=True,
                       skiprows=1)

    df = pd.concat([train, test], ignore_index=True)
    df.dropna(inplace=True)

    # Target: 1 = >50K (favorable)
    df["income"] = df["income"].str.strip().str.rstrip(".")
    y = (df["income"] == ">50K").astype(int).rename("income")

    # Sensitive attributes
    sex = (df["sex"].str.strip() == "Male").astype(int).rename("sex")
    race = (df["race"].str.strip() == "White").astype(int).rename("race")
    sensitive_attrs = pd.DataFrame({"sex": sex, "race": race})

    # Features: drop target + raw sensitive cols
    drop_cols = {"income", "sex", "race"}
    X = df.drop(columns=list(drop_cols)).copy()

    # Encode categoricals
    cat_cols = X.select_dtypes("object").columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    _print_stats("Adult", X, y, sensitive_attrs)
    return X, y, sensitive_attrs


# ---------------------------------------------------------------------------
# German Credit
# ---------------------------------------------------------------------------

_GERMAN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
)

_GERMAN_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_installment", "housing",
    "existing_credits", "job", "liable_people", "telephone", "foreign_worker", "target",
]


def load_german(
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load and preprocess the German Credit dataset.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
        Binary target: 1 = good credit (favorable).
    sensitive_attrs : pd.DataFrame
        Columns: ``sex`` (male=1/female=0), ``age`` (>=25=1/<25=0).
    """
    path = data_dir / "german.data"
    _download(_GERMAN_URL, path)

    df = pd.read_csv(path, sep=" ", names=_GERMAN_COLUMNS)

    # Target: 1 = good credit (original coding: 1=good, 2=bad)
    y = (df["target"] == 1).astype(int).rename("credit")

    # Sex derived from personal_status (A91=male divorced, A92=female divorced,
    # A93=male single, A94=male married, A95=female single)
    male_codes = {"A91", "A93", "A94"}
    sex = df["personal_status"].apply(lambda v: 1 if v in male_codes else 0).rename("sex")

    # Age binarized at 25
    age_bin = (df["age"] >= 25).astype(int).rename("age")
    sensitive_attrs = pd.DataFrame({"sex": sex, "age": age_bin})

    drop_cols = {"target", "personal_status", "age"}
    X = df.drop(columns=list(drop_cols)).copy()
    cat_cols = X.select_dtypes("object").columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    _print_stats("German Credit", X, y, sensitive_attrs)
    return X, y, sensitive_attrs


# ---------------------------------------------------------------------------
# COMPAS
# ---------------------------------------------------------------------------

_COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)


def load_compas(
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load and preprocess the ProPublica COMPAS dataset.

    NOTE: Labels are contested (Bao et al. 2021). Results should be reported
    with an explicit caveat that recidivism labels are not ground truth.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
        Binary target: 1 = no recidivism within two years (favorable).
    sensitive_attrs : pd.DataFrame
        Columns: ``race`` (Caucasian=1 / African-American=0).
    """
    path = data_dir / "compas.csv"
    _download(_COMPAS_URL, path)

    df = pd.read_csv(path)

    # Standard ProPublica filtering
    df = df[df["days_b_screening_arrest"].between(-30, 30)]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["score_text"] != "N/A"]
    df = df[df["race"].isin(["African-American", "Caucasian"])]

    # Target: 1 = no recidivism (favorable)
    y = (df["two_year_recid"] == 0).astype(int).rename("no_recid")

    # Sensitive attribute
    race = (df["race"] == "Caucasian").astype(int).rename("race")
    sensitive_attrs = pd.DataFrame({"race": race}, index=df.index)

    feature_cols = ["age", "priors_count", "juv_fel_count", "juv_misd_count",
                    "juv_other_count", "c_charge_degree", "sex"]
    X = df[feature_cols].copy()
    X["sex"] = (X["sex"] == "Male").astype(int)
    X["c_charge_degree"] = (X["c_charge_degree"] == "F").astype(int)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    sensitive_attrs = sensitive_attrs.reset_index(drop=True)

    _print_stats("COMPAS", X, y, sensitive_attrs)
    return X, y, sensitive_attrs


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _print_stats(name: str, X: pd.DataFrame, y: pd.Series,
                 sensitive_attrs: pd.DataFrame) -> None:
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"  Rows: {len(X):,}  |  Features: {X.shape[1]}")
    pos = y.mean()
    print(f"  Target balance: {pos:.3f} positive ({(y==1).sum():,} / {len(y):,})")
    for col in sensitive_attrs.columns:
        vc = sensitive_attrs[col].value_counts().to_dict()
        print(f"  {col}: {vc}")
    print(f"{'='*50}\n")
