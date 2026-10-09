import numpy as np
import pytest

from feature_schema import FEATURE_NAMES, compute_features, validate_record


def test_feature_order_is_fixed():
    assert FEATURE_NAMES == [
        "total_packets", "total_bytes", "duration_s",
        "packets_per_sec", "bytes_per_sec", "avg_packet_size",
    ]


def test_known_values():
    f = compute_features([10], [5000], [2_000_000]).iloc[0]
    assert f["duration_s"] == pytest.approx(2.0)
    assert f["packets_per_sec"] == pytest.approx(5.0)
    assert f["bytes_per_sec"] == pytest.approx(2500.0)
    assert f["avg_packet_size"] == pytest.approx(500.0)


def test_zero_duration_gives_zero_rates_not_inf():
    f = compute_features([10], [5000], [0]).iloc[0]
    assert f["packets_per_sec"] == 0.0
    assert f["bytes_per_sec"] == 0.0
    assert np.isfinite(f.to_numpy(dtype=float)).all()


def test_zero_packets_gives_zero_average():
    f = compute_features([0], [0], [1000]).iloc[0]
    assert f["avg_packet_size"] == 0.0
    assert np.isfinite(f.to_numpy(dtype=float)).all()


def test_validate_accepts_good_record():
    validate_record({"total_packets": 3, "total_bytes": 90.0, "duration_us": 1000})


@pytest.mark.parametrize("bad", [
    {"total_packets": 3, "total_bytes": 90.0},                       # missing key
    {"total_packets": 3, "total_bytes": 90.0, "duration_us": 1, "x": 1},  # extra key
    {"total_packets": float("nan"), "total_bytes": 90.0, "duration_us": 1},
    {"total_packets": float("inf"), "total_bytes": 90.0, "duration_us": 1},
    {"total_packets": -1, "total_bytes": 90.0, "duration_us": 1},
    {"total_packets": "3", "total_bytes": 90.0, "duration_us": 1},
    {"total_packets": True, "total_bytes": 90.0, "duration_us": 1},
])
def test_validate_rejects_bad_records(bad):
    with pytest.raises(ValueError):
        validate_record(bad)