"""Investigate rows where raw Flow Bytes/s differs from our recomputed value.

Read-only. Uses the first 20,000 rows of one CICIDS2017 file.
"""
import glob

import numpy as np
import pandas as pd

path = glob.glob(r"ml\data\raw\cicids2017\*\Wednesday-workingHours*.csv")[0]
header = pd.read_csv(path, nrows=0).columns
m = {c.strip(): c for c in header}

want = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
]
df = pd.read_csv(path, usecols=[m[w] for w in want], nrows=20_000,
                 dtype=str, keep_default_na=False)
df.columns = want
num = df.apply(pd.to_numeric, errors="coerce")

dur = num["Flow Duration"]
tot_bytes = num["Total Length of Fwd Packets"] + num["Total Length of Bwd Packets"]
tot_pkts = num["Total Fwd Packets"] + num["Total Backward Packets"]
raw = num["Flow Bytes/s"]

with np.errstate(divide="ignore", invalid="ignore"):
    ours = np.where(dur > 0, tot_bytes / (dur * 1e-6), 0.0)
    ratio = raw / ours

both_zero = (raw == 0) & (tot_bytes == 0)
comparable = (dur > 0) & ~both_zero & np.isfinite(raw)
ok = comparable & np.isclose(ratio, 1.0, rtol=5e-3)
bad = comparable & ~ok

print("rows:", len(df))
print("zero duration (excluded):", int((dur == 0).sum()))
print("zero bytes in both raw and recomputed (excluded):", int(both_zero.sum()))
print("comparable rows:", int(comparable.sum()))
print("match within 0.5%:", f"{ok.sum() / comparable.sum():.2%}")
print("mismatches:", int(bad.sum()))
if bad.sum():
    pct = np.percentile(ratio[bad], [1, 25, 50, 75, 99])
    print("mismatch ratio percentiles (1, 25, 50, 75, 99):", np.round(pct, 4))
    print("\nmismatches by total packet count:")
    print(tot_pkts[bad].value_counts().head(10).to_string())

    sample = pd.DataFrame({
        "dur_us": dur[bad],
        "tot_pkts": tot_pkts[bad],
        "tot_bytes": tot_bytes[bad],
        "raw_bps": raw[bad],
        "our_bps": ours[bad],
        "ratio": ratio[bad],
    })
    print("\nsample mismatches:")
    print(sample.head(10).to_string())