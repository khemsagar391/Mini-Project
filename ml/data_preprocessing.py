"""Shared loading, cleaning and time-ordered splitting for CICIDS2017.

Used by training, evaluation and tests. Does not modify raw files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Web-attack labels were stored with a bad byte; encoding_errors="replace"
# turns it into U+FFFD. Map to clean names here.
LABEL_MAP = {
    "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
    "Web Attack \ufffd XSS": "Web Attack - XSS",
    "Web Attack \ufffd Sql Injection": "Web Attack - Sql Injection",
}

TRAIN_END, VAL_END = 0.70, 0.85
TS_FORMATS = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M")

# Canonical name -> CICIDS2017 column name (after stripping spaces).
RAW_COLUMNS = {
    "timestamp": "Timestamp",
    "label": "Label",
    "duration_us": "Flow Duration",
    "total_packets_fwd": "Total Fwd Packets",
    "total_packets_bwd": "Total Backward Packets",
    "total_bytes_fwd": "Total Length of Fwd Packets",
    "total_bytes_bwd": "Total Length of Bwd Packets",
}
NUMERIC = [k for k in RAW_COLUMNS if k not in ("timestamp", "label")]


def parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse day/month/year timestamps, with or without seconds. Unparsed values stay NaT."""
    out = pd.to_datetime(series, format=TS_FORMATS[0], errors="coerce")
    miss = out.isna()
    if miss.any():
        out[miss] = pd.to_datetime(series[miss], format=TS_FORMATS[1], errors="coerce")
    return out


def load_cicids_file(path: Path) -> pd.DataFrame:
    """Load one CICIDS2017 CSV with canonical column names.

    Drops comma-only padding rows, strips text, parses timestamps and
    numbers, and maps web-attack labels. Raises on any value it cannot read.
    """
    header = pd.read_csv(path, nrows=0).columns
    stripped = {c.strip(): c for c in header}
    missing = [v for v in RAW_COLUMNS.values() if v not in stripped]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    rename = {stripped[v]: k for k, v in RAW_COLUMNS.items()}
    df = pd.read_csv(
        path,
        usecols=list(rename),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding_errors="replace",
    ).rename(columns=rename)
    df = df[list(RAW_COLUMNS)]

    for c in df.columns:
        df[c] = df[c].str.strip()

    all_empty = (df == "").all(axis=1)
    df = df.loc[~all_empty].copy()

    if (df["label"] == "").any():
        raise RuntimeError(f"{path.name}: rows with a non-padding but empty Label")

    ts = parse_timestamp(df["timestamp"])
    if ts.isna().any():
        bad = df.loc[ts.isna(), "timestamp"].head(5).tolist()
        raise RuntimeError(f"{path.name}: unparsed timestamps, e.g. {bad}")
    df["timestamp"] = ts

    for c in NUMERIC:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            bad = df.loc[v.isna(), c].head(5).tolist()
            raise RuntimeError(f"{path.name}: non-numeric {c}, e.g. {bad}")
        df[c] = v.astype("float64")

    df["label"] = df["label"].replace(LABEL_MAP)
    return df.reset_index(drop=True)


MIN_ROWS_PER_LABEL = 10  # below this, a 70/15/15 cut leaves an empty split


def time_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'split' column (train/val/test) by time order within each label.

    Each label is sorted by timestamp and cut 70/15/15. This is not a true
    session split: flows from one attack run can fall on both sides of a cut.
    Raises if a label is too small to give every split at least one row.
    """
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    split = np.full(len(df), "train", dtype=object)
    for label, idx in df.groupby("label").indices.items():
        n = len(idx)
        if n < MIN_ROWS_PER_LABEL:
            raise RuntimeError(
                f"label {label!r} has {n} rows; need at least {MIN_ROWS_PER_LABEL} to split"
            )
        c1, c2 = int(n * TRAIN_END), int(n * VAL_END)
        split[idx[c1:c2]] = "val"
        split[idx[c2:]] = "test"
    df["split"] = split
    return df