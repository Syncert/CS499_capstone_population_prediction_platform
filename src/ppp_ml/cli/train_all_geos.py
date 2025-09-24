# src/ppp_ml/cli/train_all_geos.py
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Iterable

from ppp_ml.db import list_geos, load_feature_matrix
from ppp_ml.hashers import dataframe_hash
from ppp_ml.artifacts import get_artifact_hash

MODELS: list[str] = ["prophet", "linear", "ridge", "xgb"]

def run(cmd: Iterable[str]) -> int:
    cmd = list(cmd)
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-years", type=int, default=8, help="minimum rows per geo in feature matrix")
    ap.add_argument("--horizon", type=int, default=10, help="Prophet forecast horizon (years)")
    ap.add_argument("--split-year", type=int, default=2020, help="train/test split for sklearn/xgb trainers")
    ap.add_argument("--geos", nargs="*", help="explicit list; default = query from DB")
    ap.add_argument("--models", nargs="*", choices=MODELS, help="subset of models to train")
    ap.add_argument("--skip", nargs="*", default=[], choices=MODELS, help="models to skip")
    ap.add_argument("--force", action="store_true", help="retrain regardless of stored hash")
    ap.add_argument("--dry-run", action="store_true", help="print what would train; do not run")
    args = ap.parse_args()

    selected_models = [m for m in (args.models or MODELS) if m not in set(args.skip)]

    geos = args.geos or list_geos(min_years=args.min_years, require_full=True)
    if not geos:
        print("No qualifying geos found.", file=sys.stderr)
        sys.exit(2)
    if not selected_models:
        print("Nothing to do: no models selected.", file=sys.stderr)
        sys.exit(2)

    # Build todo list with drift check
    todo: list[tuple[str, str]] = []
    for g in geos:
        df = load_feature_matrix(g)
        if df.empty:
            continue
        live = dataframe_hash(df)
        for m in selected_models:
            prev = get_artifact_hash(g, m)  # None if missing
            if args.force or prev != live:
                todo.append((g, m))

    if not todo:
        print("Nothing to train — all up to date ✅")
        return

    print(f"{len(todo)} trainings to run:")
    for g, m in todo:
        print(f"  - {m} @ {g}")

    if args.dry_run:
        return

    failed: list[tuple[str, str]] = []
    for g, m in todo:
        if m == "prophet":
            rc = run([sys.executable, "-m", "ppp_ml.cli.train_prophet", "--geo", g,
                      "--horizon", str(args.horizon)])
        elif m == "linear":
            rc = run([sys.executable, "-m", "ppp_ml.cli.train_linear", "--geo", g,
                      "--split_year", str(args.split_year)])
        elif m == "ridge":
            rc = run([sys.executable, "-m", "ppp_ml.cli.train_ridge", "--geo", g,
                      "--split_year", str(args.split_year)])
        else:  # xgb
            rc = run([sys.executable, "-m", "ppp_ml.cli.train_xgb", "--geo", g,
                      "--split_year", str(args.split_year)])
        if rc != 0:
            failed.append((g, m))

    if failed:
        print("FAILED:", failed, file=sys.stderr)
        sys.exit(1)

    print("Training complete ✅")

if __name__ == "__main__":
    main()