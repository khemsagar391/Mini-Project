"""Reload the saved Isolation Forest in a clean process and compare outputs.

Pass criteria: unusualness scores match to 1e-9 and the alarm decisions are identical.
"""
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]  # the ml/ folder
HERE = ML
sys.path[:0] = [str(ML), str(ML / "models"), str(ML / "reports")]

import sys
from pathlib import Path

import joblib
import numpy as np

import sys
from pathlib import Path

MODELS = HERE / "models"
NAME = "isolation_forest_v1"


def main():
    bundle = joblib.load(MODELS / f"{NAME}.joblib")
    model, threshold = bundle["model"], bundle["threshold"]
    ref = np.load(MODELS / f"{NAME}_test_reference.npz")

    unusual = -model.score_samples(ref["X"])
    flags = unusual > threshold

    max_diff = float(np.max(np.abs(unusual - ref["scores"])))
    same_flags = np.array_equal(flags, ref["flags"])
    print(f"rows checked: {len(unusual):,}")
    print(f"largest score difference: {max_diff:.2e}")
    print(f"alarm decisions identical: {same_flags}")
    print(f"threshold identical: {float(threshold) == float(ref['threshold'])}")

    if max_diff < 1e-9 and same_flags:
        print("RELOAD CHECK PASSED")
        return 0
    print("RELOAD CHECK FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())