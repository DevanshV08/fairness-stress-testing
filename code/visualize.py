"""Generate project figures from CSV/JSON outputs in results/."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {path}")


def _maybe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def load_grid_results(path: Path = RESULTS_DIR / "grid_results.csv") -> pd.DataFrame:
    df = _maybe_read_csv(path)
    if df is None:
        return pd.DataFrame()
    return df


def load_pattern(pattern: str) -> pd.DataFrame:
    frames = []
    for path in RESULTS_DIR.glob(pattern):
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        metadata = _metadata_from_name(path.stem)
        for key, value in metadata.items():
            if key not in df.columns:
                df[key] = value
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _metadata_from_name(stem: str) -> dict[str, str]:
    parts = stem.split("_")
    if stem.startswith("seed_sweep_") and len(parts) >= 6:
        return {
            "dataset": parts[2],
            "model": parts[3],
            "debias_method": "_".join(parts[4:-1]),
            "sensitive_col": parts[-1],
        }
    if stem.startswith("threshold_sweep_") and len(parts) >= 7:
        return {
            "dataset": parts[2],
            "model": parts[3],
            "debias_method": "_".join(parts[4:-2]),
            "sensitive_col": parts[-2],
            "seed": parts[-1].removeprefix("seed"),
        }
    if stem.startswith("intersectional_") and len(parts) >= 5:
        return {
            "dataset": parts[1],
            "model": parts[2],
            "debias_method": "_".join(parts[3:-1]),
            "seed": parts[-1].removeprefix("seed"),
        }
    return {}


def plot_baseline_overview(grid_df: pd.DataFrame) -> None:
    df = grid_df[(grid_df["debias_method"] == "none") & grid_df["accuracy"].notna()].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=df, x="dataset", y="accuracy", hue="model", ax=axes[0])
    axes[0].set_title("Baseline Accuracy by Dataset")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("")

    sns.barplot(data=df, x="dataset", y="dp_diff", hue="model", ax=axes[1])
    axes[1].set_title("Baseline DP Difference by Dataset")
    axes[1].set_xlabel("")
    axes[1].axhline(0, color="black", linewidth=1)

    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend_.remove()
    axes[1].legend(handles, labels, title="model", loc="best")
    _save(fig, "baseline_overview.png")


def plot_adult_method_tradeoff(grid_df: pd.DataFrame) -> None:
    df = grid_df[
        (grid_df["dataset"] == "adult")
        & grid_df["accuracy"].notna()
        & grid_df["dp_diff"].notna()
    ].copy()
    if df.empty:
        return

    summary = (
        df.groupby(["model", "debias_method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            dp_mean=("dp_diff", "mean"),
            dp_std=("dp_diff", "std"),
        )
        .fillna(0.0)
    )
    summary["abs_dp_mean"] = summary["dp_mean"].abs()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(data=summary, x="debias_method", y="accuracy_mean", hue="model", ax=axes[0])
    axes[0].set_title("Adult Accuracy by Debiasing Method")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylim(0, 1)

    sns.barplot(data=summary, x="debias_method", y="abs_dp_mean", hue="model", ax=axes[1])
    axes[1].set_title("Adult |DP Difference| by Debiasing Method")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)

    axes[0].legend_.remove()
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, title="model", loc="best")
    _save(fig, "adult_method_tradeoff.png")


def plot_pareto_front(grid_df: pd.DataFrame) -> None:
    df = grid_df[
        (grid_df["dataset"] == "adult")
        & grid_df["accuracy"].notna()
        & grid_df["dp_diff"].notna()
    ].copy()
    if df.empty:
        return

    df["abs_dp_diff"] = df["dp_diff"].abs()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="abs_dp_diff",
        y="accuracy",
        hue="model",
        style="debias_method",
        s=90,
        ax=ax,
    )
    ax.set_title("Adult Pareto Front: Accuracy vs |DP Difference|")
    ax.set_xlabel("|Demographic Parity Difference|")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(left=0)
    ax.set_ylim(0.75, 0.9)
    _save(fig, "adult_pareto_front.png")


def plot_seed_sweep(seed_df: pd.DataFrame) -> None:
    if seed_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.lineplot(
        data=seed_df,
        x="seed",
        y="accuracy",
        hue="debias_method",
        style="model",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Seed Sweep Accuracy")
    axes[0].set_ylim(0, 1)

    seed_df = seed_df.copy()
    seed_df["abs_dp_diff"] = seed_df["dp_diff"].abs()
    sns.lineplot(
        data=seed_df,
        x="seed",
        y="abs_dp_diff",
        hue="debias_method",
        style="model",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("Seed Sweep |DP Difference|")
    axes[1].set_ylabel("|DP Difference|")

    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    _save(fig, "seed_sweep.png")


def plot_threshold_sweep(threshold_df: pd.DataFrame) -> None:
    if threshold_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.lineplot(
        data=threshold_df,
        x="threshold",
        y="accuracy",
        hue="model",
        style="debias_method",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Threshold Sweep Accuracy")
    axes[0].set_ylim(0, 1)

    threshold_df = threshold_df.copy()
    threshold_df["abs_dp_diff"] = threshold_df["dp_diff"].abs()
    sns.lineplot(
        data=threshold_df,
        x="threshold",
        y="abs_dp_diff",
        hue="model",
        style="debias_method",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("Threshold Sweep |DP Difference|")
    axes[1].set_ylabel("|DP Difference|")

    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    _save(fig, "threshold_sweep.png")


def plot_intersectional(intersectional_df: pd.DataFrame) -> None:
    if intersectional_df.empty:
        return

    df = intersectional_df.copy()
    pair_cols = [c for c in ["sex", "race", "age"] if c in df.columns]
    if len(pair_cols) < 2:
        return
    index_col, col_col = pair_cols[0], pair_cols[1]
    subset = df[(df["model"] == df["model"].iloc[0]) & (df["debias_method"] == df["debias_method"].iloc[0])]
    pivot = subset.pivot(index=index_col, columns=col_col, values="selection_rate")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title("Intersectional Selection Rates")
    _save(fig, "intersectional_selection_rate.png")


def write_summary(grid_df: pd.DataFrame) -> None:
    if grid_df.empty:
        return
    summary = (
        grid_df.groupby(["dataset", "model", "debias_method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            dp_mean=("dp_diff", "mean"),
            dp_std=("dp_diff", "std"),
            eo_mean=("eo_diff", "mean"),
            eod_mean=("eod_diff", "mean"),
        )
        .fillna(0.0)
    )
    out = RESULTS_DIR / "summary_metrics.csv"
    summary.to_csv(out, index=False)
    print(f"Saved summary -> {out}")


def build_figures() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    grid_df = load_grid_results()
    seed_df = load_pattern("seed_sweep_*.csv")
    threshold_df = load_pattern("threshold_sweep_*.csv")
    intersectional_df = load_pattern("intersectional_*.csv")

    plot_baseline_overview(grid_df)
    plot_adult_method_tradeoff(grid_df)
    plot_pareto_front(grid_df)
    plot_seed_sweep(seed_df)
    plot_threshold_sweep(threshold_df)
    plot_intersectional(intersectional_df)
    write_summary(grid_df)


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures from experiment results.")
    return parser.parse_args(args)


def main() -> None:
    parse_args()
    build_figures()


if __name__ == "__main__":
    main()
