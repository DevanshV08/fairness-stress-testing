"""CLI entry point for running experiments.

Usage examples
--------------
# Full default grid (5 seeds, all datasets/models/methods)
python run_experiments.py

# Quick sanity check: Adult, LR baseline, seed 0 only
python run_experiments.py --datasets adult --models lr --methods none --seeds 0

# Stress tests only
python run_experiments.py --mode stress --datasets adult --models lr xgboost

# Single run
python run_experiments.py --mode single --datasets adult --models lr --methods reweighing --seeds 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

# Ensure code/ is on path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import run_single, run_grid, LOADERS
from stress_tests import run_seed_sweep, run_threshold_sweep, run_intersectional
from debiasing import DEBIAS_METHODS


DEFAULT_DATASETS = ["adult", "german", "compas"]
DEFAULT_MODELS = ["lr", "rf", "xgboost", "ft_transformer"]
DEFAULT_METHODS = list(DEBIAS_METHODS)
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fairness stress-test experiment runner")
    p.add_argument("--mode", choices=["grid", "stress", "single"], default="grid",
                   help="grid=full cross-product; stress=seed/threshold/intersectional; "
                        "single=one run")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--no-save", action="store_true", help="Do not write results to disk")
    return p.parse_args(argv)


def run_stress_suite(args: argparse.Namespace) -> None:
    save = not args.no_save
    for ds in args.datasets:
        for mdl in args.models:
            for mth in args.methods:
                print(f"\n--- Seed sweep: {ds} | {mdl} | {mth} ---")
                df = run_seed_sweep(ds, mdl, mth, seeds=args.seeds, save=save)
                if not df.empty:
                    mean = df[["accuracy", "dp_diff", "eo_diff"]].mean()
                    std = df[["accuracy", "dp_diff", "eo_diff"]].std()
                    print(df[["seed", "accuracy", "dp_diff", "eo_diff"]].to_string(index=False))
                    print(f"  mean: {mean.to_dict()}")
                    print(f"  std:  {std.to_dict()}")

                print(f"\n--- Threshold sweep: {ds} | {mdl} | {mth} ---")
                df_t = run_threshold_sweep(ds, mdl, debias_method=mth, seed=args.seeds[0], save=save)
                if not df_t.empty:
                    print(df_t.to_string(index=False))

                print(f"\n--- Intersectional: {ds} | {mdl} | {mth} ---")
                df_i = run_intersectional(ds, mdl, debias_method=mth, seed=args.seeds[0], save=save)
                if not df_i.empty:
                    print(df_i.to_string(index=False))


def main(argv: List[str] = None) -> None:
    args = parse_args(argv)
    save = not args.no_save
    t0 = time.time()

    if args.mode == "single":
        assert len(args.datasets) == 1 and len(args.models) == 1 and len(args.methods) == 1, \
            "single mode requires exactly one dataset, model, and method"
        result = run_single(
            args.datasets[0], args.models[0], args.methods[0],
            seed=args.seeds[0], save=save
        )
        print(result)

    elif args.mode == "grid":
        print(f"Grid: {len(args.datasets)} datasets x {len(args.models)} models x "
              f"{len(args.methods)} methods x {len(args.seeds)} seeds")
        total = (len(args.datasets) * len(args.models)
                 * len(args.methods) * len(args.seeds))
        print(f"Total runs: {total}  (estimated ~{total * 1.5:.0f}s on RTX 3060)\n")
        df = run_grid(args.datasets, args.models, args.methods, args.seeds,
                      save=save)
        print(f"\nGrid complete. Shape: {df.shape}")
        numeric_cols = [c for c in ["accuracy", "dp_diff", "eo_diff", "eod_diff"]
                        if c in df.columns]
        if numeric_cols:
            print(df.groupby(["dataset", "model", "debias_method"])[numeric_cols]
                  .agg(["mean", "std"]).round(4))

    elif args.mode == "stress":
        run_stress_suite(args)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
