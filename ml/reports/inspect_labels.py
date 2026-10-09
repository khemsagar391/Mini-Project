from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent / "data" / "raw"
OUT = Path(__file__).resolve().parent / "reports" / "label_counts.csv"

FILES = sorted(
    list(ROOT.glob("cicids2017/*/*.csv"))
    + list(ROOT.glob("insdn/InSDN_DatasetCSV/*.csv"))
)


def find_label_col(path: Path) -> str:
    header = pd.read_csv(path, nrows=0)
    matches = [c for c in header.columns if c.strip() == "Label"]
    if len(matches) != 1:
        raise ValueError(f"{path.name}: expected one Label column, found {matches}")
    return matches[0]


def count_labels(path: Path):
    """Count raw label strings. Blank labels (comma-only padding rows) are counted separately."""
    label_col = find_label_col(path)
    counts = Counter()
    rows = 0
    padding = 0
    reader = pd.read_csv(
        path,
        usecols=[label_col],
        dtype=str,
        keep_default_na=False,      # keep "NA", "", etc. as literal text
        na_filter=False,            # do not convert anything to NaN
        chunksize=500_000,
        encoding_errors="replace",  # CICIDS2017 web-attack labels contain bad bytes
    )
    for chunk in reader:
        labels = chunk[label_col].str.strip()
        blank = int((labels == "").sum())
        padding += blank
        labels = labels[labels != ""]  # drop padding rows
        counts.update(labels.value_counts().to_dict())
        rows += len(labels)
    if sum(counts.values()) != rows:
        raise RuntimeError(f"{path.name}: counted {sum(counts.values())} of {rows} rows")
    return counts, rows, padding


def main() -> None:
    if not FILES:
        raise SystemExit(f"No CSV files found under {ROOT}")

    rows = []
    for path in FILES:
        dataset = "CICIDS2017" if "cicids2017" in path.parts else "InSDN"
        print(f"Reading {path.name} ...", flush=True)
        counts, total, padding = count_labels(path)
        print(f"  {total:,} rows, {len(counts)} distinct labels, {padding} padding dropped")
        for label, n in counts.most_common():
            rows.append({"dataset": dataset, "file": path.name, "label": label, "count": n})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)

    print("\nTotals per label across all files:")
    print(result.groupby(["dataset", "label"])["count"].sum().to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()