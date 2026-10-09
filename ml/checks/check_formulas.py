"""Check bytes/s, average packet size, and zero-duration rows against raw columns. Read-only."""
import glob

import numpy as np
import pandas as pd

N = 20_000
CASES = {
    "CICIDS2017": (
        glob.glob(r"ml\data\raw\cicids2017\*\Wednesday-workingHours*.csv")[0],
        {"dur": "Flow Duration", "fwd": "Total Fwd Packets", "bwd": "Total Backward Packets",
         "fb": "Total Length of Fwd Packets", "bb": "Total Length of Bwd Packets",
         "bps": "Flow Bytes/s", "aps": "Average Packet Size"},
    ),
    "InSDN": (
        glob.glob(r"ml\data\raw\insdn\InSDN_DatasetCSV\Normal_data.csv")[0],
        {"dur": "Flow Duration", "fwd": "Tot Fwd Pkts", "bwd": "Tot Bwd Pkts",
         "fb": "TotLen Fwd Pkts", "bb": "TotLen Bwd Pkts",
         "bps": "Flow Byts/s", "aps": "Pkt Size Avg"},
    ),
}

for name, (path, c) in CASES.items():
    raw = pd.read_csv(path, nrows=0).columns
    m = {col.strip(): col for col in raw}
    wanted = list(c.values())
    df = pd.read_csv(path, usecols=[m[w] for w in wanted], nrows=N, low_memory=False)
    df.columns = wanted
    df = df.apply(pd.to_numeric, errors="coerce")

    dur, tot_p = df[c["dur"]], df[c["fwd"]] + df[c["bwd"]]
    tot_b = df[c["fb"]] + df[c["bb"]]
    pos = dur > 0

    calc_bps = tot_b[pos] / (dur[pos] * 1e-6)
    ratio_bps = df.loc[pos, c["bps"]] / calc_bps
    close_bps = np.isclose(ratio_bps, 1.0, rtol=1e-6).mean()

    pos_p = tot_p > 0
    calc_aps = tot_b[pos_p] / tot_p[pos_p]
    ratio_aps = df.loc[pos_p, c["aps"]] / calc_aps
    close_aps = np.isclose(ratio_aps, 1.0, rtol=1e-6).mean()

    print(f"{name}: sample rows = {len(df)}")
    print(f"  zero-duration rows: {(~pos).sum()}")
    print(f"  bytes/s: median ratio {ratio_bps.median():.6f}, within 1e-6 of raw: {close_bps:.2%}")
    print(f"  avg packet size: median ratio {ratio_aps.median():.6f}, within 1e-6 of raw: {close_aps:.2%}")