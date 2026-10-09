"""Feature contract for the RouteSense ML module (version 1.0).

Training, evaluation and the live predictor must all use compute_features().
Do not re-derive features anywhere else.
"""
import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"

# Model input order. Do not reorder without bumping SCHEMA_VERSION.
FEATURE_NAMES = [
    "total_packets",
    "total_bytes",
    "duration_s",
    "packets_per_sec",
    "bytes_per_sec",
    "avg_packet_size",
]

# Counters the controller must supply for each flow (both directions summed).
RAW_INPUTS = ("total_packets", "total_bytes", "duration_us")


def compute_features(total_packets, total_bytes, duration_us) -> pd.DataFrame:
    """Return one row per flow with FEATURE_NAMES in order.

    Zero-denominator rule: a rate is 0 when duration is 0;
    the average packet size is 0 when there are no packets.
    """
    pk = np.atleast_1d(np.asarray(total_packets, dtype=float))
    by = np.atleast_1d(np.asarray(total_bytes, dtype=float))
    du = np.atleast_1d(np.asarray(duration_us, dtype=float))

    dur_s = du / 1e6

    pps = np.zeros_like(pk)
    np.divide(pk, dur_s, out=pps, where=dur_s > 0)

    bps = np.zeros_like(by)
    np.divide(by, dur_s, out=bps, where=dur_s > 0)

    aps = np.zeros_like(by)
    np.divide(by, pk, out=aps, where=pk > 0)

    return pd.DataFrame(
        {
            "total_packets": pk,
            "total_bytes": by,
            "duration_s": dur_s,
            "packets_per_sec": pps,
            "bytes_per_sec": bps,
            "avg_packet_size": aps,
        }
    )[FEATURE_NAMES]


def validate_record(record) -> None:
    """Reject a live feature record that is malformed. Raises ValueError.

    Checks exact key set, numeric type, finiteness and non-negativity.
    Used by the controller interface before any prediction.
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    keys = set(record)
    if keys != set(RAW_INPUTS):
        raise ValueError(
            f"record keys must be exactly {sorted(RAW_INPUTS)}, got {sorted(keys)}"
        )
    for name in RAW_INPUTS:
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name} must be a number, got {type(value).__name__}")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")