from __future__ import annotations
import argparse, subprocess, sys
from ppp_ml.db import list_geos

def run(cmd: list[str]) -> int:
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-years", type=int, default=8, help="minimum rows per geo in ml.feature_matrix")
    ap.add_argument("--horizon", type=int, default=10, help="prophet forecast horizon (years)")
    ap.add_argument("--geos", nargs="*", help="optional explicit list; default = query from DB")
    ap.add_argument("--skip", nargs="*", default=[], choices=["prophet","linear","ridge","xgb"], help="skip certain trainers")
    ap.add_argument("--split-year", type=int, default=2020, help="train/test split for sklearn/xgb trainers")
    args = ap.parse_args()

    geos = args.geos or list_geos(min_years=args.min_years, require_full=True)
    if not geos:
        print("No qualifying geos found.", file=sys.stderr)
        sys.exit(2)

    failed: list[str] = []
    for g in geos:
        rc = 0
        if "prophet" not in args.skip:
            rc |= run([sys.executable, "-m", "ppp_ml.cli.train_prophet", "--geo", g, "--horizon", str(args.horizon)])
        if "linear" not in args.skip:
            rc |= run([sys.executable, "-m", "ppp_ml.cli.train_linear", "--geo", g, "--split_year", str(args.split_year)])
        if "ridge" not in args.skip:
            rc |= run([sys.executable, "-m", "ppp_ml.cli.train_ridge",  "--geo", g, "--split_year", str(args.split_year)])
        if "xgb" not in args.skip:
            rc |= run([sys.executable, "-m", "ppp_ml.cli.train_xgb",    "--geo", g, "--split_year", str(args.split_year)])

        if rc != 0:
            failed.append(g)

    if failed:
        print(f"FAILED geos: {failed}", file=sys.stderr)
        sys.exit(1)

    print("All geos trained")

if __name__ == "__main__":
    main()