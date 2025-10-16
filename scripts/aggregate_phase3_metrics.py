import argparse
import pathlib
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="results/ml/feat-ml-phase3")
    ap.add_argument(
        "--output", default="results/ml/feat-ml-phase3/aggregate_summary.csv"
    )
    args = ap.parse_args()

    rows = []
    for p in pathlib.Path(args.input_root).rglob("*_metrics.csv"):
        try:
            df = pd.read_csv(p)
            df["run"] = p.stem
            rows.append(df)
        except Exception:
            pass
    if not rows:
        raise SystemExit("No *_metrics.csv files found.")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(args.output, index=False)
    print(f"[aggregate] wrote {args.output} with {len(out)} rows")


if __name__ == "__main__":
    main()
