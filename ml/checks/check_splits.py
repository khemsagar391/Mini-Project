"""Time-ordered 70/15/15 split within each CICIDS2017 file. Read-only.

Rows are sorted by Timestamp inside each file and cut into contiguous
blocks: first 70% train, next 15% validation, last 15% test.
Writes ml/reports/split_counts.csv. Does not modify the raw data.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_preprocessing import LABEL_MAP  # same clean names as the loader

FILES = sorted(glob.glob(str(HERE / "data" / "raw" / "cicids2017" / "*" / "*.csv")))
OUT = HERE / "reports" / "split_counts.csv"
TS_FORMAT = "%d/%m/%Y %H:%M"   # verified format: "6/7/2017 12:59" = day/month/year
TRAIN_END, VAL_END = 0.70, 0.85


def load(path):
    header = pd.read_csv(path, nrows=0).columns
    m = {c.strip(): c for c in header}
    df = pd.read_csv(
        path,
        usecols=[m["Timestamp"], m["Label"]],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding_errors="replace",
    )
    df.columns = ["Timestamp", "Label"]
    df["Timestamp"] = df["Timestamp"].str.strip()
    df["Label"] = df["Label"].str.strip()

    # Same padding rule as the loader: drop rows where every field is empty.
    df = df[~((df["Timestamp"] == "") & (df["Label"] == ""))].copy()

    ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    fallback = ts.isna()
    ts[fallback] = pd.to_datetime(df.loc[fallback, "Timestamp"], format="%d/%m/%Y %H:%M", errors="coerce")
    bad = ts.isna()
    if bad.any():
        examples = df.loc[bad, "Timestamp"].head(5).tolist()
        raise RuntimeError(f"{path.name}: {int(bad.sum())} timestamps failed to parse, e.g. {examples}")
    df["ts"] = ts
    df["Label"] = df["Label"].replace(LABEL_MAP)
    return df


def assign_splits(df):
    """Sort by time, then split each label 70/15/15 in time order."""
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["split"] = "train"
    for label, idx in df.groupby("Label").indices.items():
        n = len(idx)
        cut1 = int(n * TRAIN_END)
        cut2 = int(n * VAL_END)
        df.loc[idx[cut1:cut2], "split"] = "val"
        df.loc[idx[cut2:], "split"] = "test"
    return df


def main():
    if not FILES:
        raise SystemExit("No CICIDS2017 CSVs found")

    parts = []
    for p in FILES:
        path = Path(p)
        df = assign_splits(load(path))
        df["file"] = path.name
        print(f"{path.name}: {len(df):,} rows, "
              f"{df['ts'].min()} to {df['ts'].max()}")
        parts.append(df[["file", "split", "Label"]])

    allrows = pd.concat(parts, ignore_index=True)
    table = pd.crosstab(allrows["Label"], allrows["split"])
    table = table.reindex(columns=["train", "val", "test"], fill_value=0)

    print("\nRows per label and split:")
    print(table.to_string())

    missing = table[(table == 0).any(axis=1)]
    print("\nLabels with zero rows in at least one split:")
    print(missing.to_string() if not missing.empty else "  none")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()