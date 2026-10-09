import sys
from pathlib import Path

import joblib
import numpy as np

import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]  # the ml/ folder
HERE = ML
sys.path[:0] = [str(ML), str(ML / "models"), str(ML / "reports")]
MODELS = HERE / "models"
NAME = "random_forest_v1"


def main():
    bundle = joblib.load(MODELS / f"{NAME}.joblib")
    model = bundle["model"]
    saved = np.load(MODELS / f"{NAME}_test_predictions.npz")

    pred = model.predict(saved["X"]).astype(int)
    proba = model.predict_proba(saved["X"])

    same_class = np.array_equal(pred, saved["pred"])
    max_diff = float(np.max(np.abs(proba - saved["proba"])))
    print(f"rows checked: {len(pred):,}")
    print(f"predicted classes identical: {same_class}")
    print(f"largest probability difference: {max_diff:.2e}")

    if same_class and max_diff < 1e-9:
        print("RELOAD CHECK PASSED")
        return 0
    print("RELOAD CHECK FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())