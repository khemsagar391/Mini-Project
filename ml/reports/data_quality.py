"""Read-only data-quality audit for CICIDS2017 and InSDN.

Reads in chunks of 100,000 rows. Writes three reports to ml/reports/:
  data_quality_files.csv    per file: rows, padding, within-file duplicates
  data_quality_columns.csv  per file and feature: inf, empty, non-numeric cells
  data_quality_datasets.csv per dataset: duplicates across files, label conflicts
Does not modify the raw data.
"""
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]  # the ml/ folder
HERE = ML
sys.path[:0] = [str(ML), str(ML / "models"), str(ML / "reports")]

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path

ROOT = HERE / "data" / "raw"
REPORTS = HERE / "reports"
CHUNK = 100_000
LABEL = "Label"


def discover():
    return sorted(
        list(ROOT.glob("cicids2017/*/*.csv"))
        + list(ROOT.glob("insdn/InSDN_DatasetCSV/*.csv"))
    )


def dataset_of(path: Path) -> str:
    return "CICIDS2017" if "cicids2017" in path.parts else "InSDN"


def main() -> None:
    files = discover()
    if not files:
        raise SystemExit(f"No CSV files found under {ROOT}")

    # Map stripped column name -> raw column name, for every file.
    headers = {}
    for p in files:
        raw = pd.read_csv(p, nrows=0, encoding_errors="replace").columns
        headers[p] = {c.strip(): c for c in raw}

    file_rows, col_rows, dataset_rows = [], [], []

    for dataset in ("CICIDS2017", "InSDN"):
        dfiles = [p for p in files if dataset_of(p) == dataset]
        if not dfiles:
            continue

        # Only compare columns present in every file of this dataset.
        first = list(headers[dfiles[0]])
        common = [c for c in first if all(c in headers[p] for p in dfiles)]
        if LABEL not in common:
            raise RuntimeError(f"{dataset}: no Label column in common across files")
        feat = [c for c in common if c != LABEL]
        print(f"{dataset}: {len(dfiles)} files, {len(common)} common columns")

        full_all, feat_all, label_all = [], [], []
        within_full_total = 0

        for p in dfiles:
            print(f"  Reading {p.name} ...", flush=True)
            usecols = [headers[p][c] for c in common]
            reader = pd.read_csv(
                p,
                usecols=usecols,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                chunksize=CHUNK,
                encoding_errors="replace",
            )

            stats = defaultdict(int)  # (column, kind) -> count
            rows_read = padding = 0
            full_parts, feat_parts, label_parts = [], [], []

            for chunk in reader:
                chunk.columns = common
                rows_read += len(chunk)

                blank = chunk.eq("").all(axis=1)  # comma-only padding rows
                padding += int(blank.sum())
                chunk = chunk.loc[~blank]
                if chunk.empty:
                    continue

                for c in feat:
                    s = chunk[c]
                    num = pd.to_numeric(s, errors="coerce")
                    stats[(c, "empty")] += int((s == "").sum())
                    stats[(c, "non_numeric")] += int((num.isna() & (s != "")).sum())
                    stats[(c, "inf_pos")] += int((num == np.inf).sum())
                    stats[(c, "inf_neg")] += int((num == -np.inf).sum())

                full_parts.append(
                    pd.util.hash_pandas_object(chunk, index=False).to_numpy(np.uint64)
                )
                feat_parts.append(
                    pd.util.hash_pandas_object(chunk[feat], index=False).to_numpy(np.uint64)
                )
                label_parts.append(chunk[LABEL].str.strip().to_numpy())

            full = np.concatenate(full_parts) if full_parts else np.array([], np.uint64)
            fh = np.concatenate(feat_parts) if feat_parts else np.array([], np.uint64)
            lab = np.concatenate(label_parts) if label_parts else np.array([], object)

            within_full = len(full) - len(np.unique(full))
            within_feat = len(fh) - len(np.unique(fh))
            within_full_total += within_full

            full_all.append(full)
            feat_all.append(fh)
            label_all.append(lab)

            file_rows.append({
                "dataset": dataset,
                "file": p.name,
                "rows_read": rows_read,
                "padding_rows_dropped": padding,
                "rows_kept": len(full),
                "within_file_exact_duplicates": within_full,
                "within_file_duplicates_ignoring_label": within_feat,
            })
            for c in feat:
                col_rows.append({
                    "dataset": dataset,
                    "file": p.name,
                    "column": c,
                    "empty_cells": stats[(c, "empty")],
                    "non_numeric_cells": stats[(c, "non_numeric")],
                    "inf_pos": stats[(c, "inf_pos")],
                    "inf_neg": stats[(c, "inf_neg")],
                })

        # Dataset-wide checks across all files.
        all_full = np.concatenate(full_all)
        all_feat = np.concatenate(feat_all)
        all_lab = np.concatenate(label_all)
        total = len(all_full)

        exact_total = total - len(np.unique(all_full))
        cross_file = exact_total - within_full_total  # extra copies that span files
        feat_total = total - len(np.unique(all_feat))

        lab_df = pd.DataFrame({"f": all_feat, "l": all_lab})
        in_conflict = lab_df.groupby("f")["l"].transform("nunique") > 1

        dataset_rows.append({
            "dataset": dataset,
            "rows_kept": total,
            "exact_duplicate_rows": exact_total,
            "exact_duplicates_within_files": within_full_total,
            "extra_copies_across_files": cross_file,
            "duplicates_ignoring_label": feat_total,
            "rows_in_label_conflict_groups": int(in_conflict.sum()),
        })

    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(file_rows).to_csv(REPORTS / "data_quality_files.csv", index=False)
    pd.DataFrame(col_rows).to_csv(REPORTS / "data_quality_columns.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(REPORTS / "data_quality_datasets.csv", index=False)

    print("\nPer dataset:")
    print(pd.DataFrame(dataset_rows).T.to_string())
    print("\nInfinite values per file (feature columns with any inf):")
    cols = pd.DataFrame(col_rows)
    bad = cols[(cols["inf_pos"] + cols["inf_neg"]) > 0]
    print(bad.to_string(index=False) if not bad.empty else "  none")
    print(f"\nReports saved to {REPORTS}")


if __name__ == "__main__":
    main()