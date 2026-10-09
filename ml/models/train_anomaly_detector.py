import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]  # the ml/ folder
HERE = ML
sys.path[:0] = [str(ML), str(ML / "models"), str(ML / "reports")]

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from feature_schema import FEATURE_NAMES, SCHEMA_VERSION
from train_classifiers import SEED, EVAL_CAP, cap_per_class, load_all, to_xy

import sys
from pathlib import Path

METRICS = HERE / "metrics"
MODELS = HERE / "models"
NAME = "isolation_forest_v1"
TARGET_FPR = 0.01  # about 1% of normal validation flows may be flagged
PARAMS = dict(n_estimators=200, max_samples=256, random_state=SEED, n_jobs=-1)


def main():
    METRICS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("Loading CICIDS2017 ...", flush=True)
    df = load_all()
    enc = LabelEncoder().fit(df["label"])
    benign_code = int(enc.transform(["BENIGN"])[0])

    # Training: normal traffic only.
    train = df[df["split"] == "train"]
    X_normal, _ = to_xy(train[train["label"] == "BENIGN"], enc)

    # Validation (threshold) and test (reporting) sets, same rows as Stage 1.
    val_eval = cap_per_class(df[df["split"] == "val"], EVAL_CAP, SEED)
    test_eval = cap_per_class(df[df["split"] == "test"], EVAL_CAP, SEED)
    X_val, y_val = to_xy(val_eval, enc)
    X_test, y_test = to_xy(test_eval, enc)
    labels_test = test_eval["label"].to_numpy()
    print(f"Training on {len(X_normal):,} normal flows", flush=True)

    t0 = time.perf_counter()
    model = IsolationForest(**PARAMS).fit(X_normal)
    fit_s = time.perf_counter() - t0
    print(f"Fit took {fit_s:.1f} s", flush=True)

    # score_samples: higher means more normal. Flip it so higher means more unusual.
    unusual_val = -model.score_samples(X_val)
    unusual_test = -model.score_samples(X_test)

    # Threshold chosen on VALIDATION normal traffic only.
    normal_val_scores = unusual_val[y_val == benign_code]
    threshold = float(np.quantile(normal_val_scores, 1 - TARGET_FPR))
    flagged_val = unusual_val > threshold
    flagged_test = unusual_test > threshold

    # Random Forest results on the same test rows, saved in Phase 3.
    saved = np.load(MODELS / "random_forest_v1_test_predictions.npz")
    if not np.array_equal(saved["X"], X_test):
        raise RuntimeError("Test rows differ from the saved Random Forest run")
    rf_pred = saved["pred"].astype(int)

    rows = []
    for label in sorted(set(labels_test)):
        m = labels_test == label
        rows.append({
            "class": label,
            "test_flows": int(m.sum()),
            "flagged_by_isolation_forest": round(float(flagged_test[m].mean()), 4),
            "found_by_random_forest": round(float((rf_pred[m] == y_test[m]).mean()), 4),
        })
    per_class = pd.DataFrame(rows)

    attack = y_test != benign_code
    summary = {
        "model_name": NAME,
        "algorithm": "sklearn IsolationForest",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "feature_schema_version": SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "training_rows": int(len(X_normal)),
        "training_labels": ["BENIGN only"],
        "params": PARAMS,
        "fit_seconds": round(fit_s, 1),
        "threshold_unusual_score": round(threshold, 6),
        "target_false_alarm_rate": TARGET_FPR,
        "validation_false_alarm_rate": round(float(flagged_val[y_val == benign_code].mean()), 4),
        "test_false_alarm_rate": round(float(flagged_test[~attack].mean()), 4),
        "test_attacks_flagged_overall": round(float(flagged_test[attack].mean()), 4),
        "versions": {"python": "3.12.10", "scikit-learn": sklearn.__version__},
    }

    joblib.dump({"model": model, "threshold": threshold,
                 "feature_names": FEATURE_NAMES}, MODELS / f"{NAME}.joblib", compress=3)
    (MODELS / f"{NAME}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_class.to_csv(METRICS / "stage2_per_class_test.csv", index=False)
    np.savez(MODELS / f"{NAME}_test_reference.npz",
             X=X_test, scores=unusual_test, flags=flagged_test, threshold=threshold)

    print("\nFalse alarm rate on validation normal traffic: "
          f"{summary['validation_false_alarm_rate'] * 100:.1f}% (target 1%)")
    print("False alarm rate on test normal traffic:      "
          f"{summary['test_false_alarm_rate'] * 100:.1f}%")
    print("Attacks flagged overall on test:               "
          f"{summary['test_attacks_flagged_overall'] * 100:.1f}%")
    print("\nPer class (test data):")
    print(per_class.to_string(index=False))
    print("\nFor BENIGN, 'found_by_random_forest' is the share it correctly left alone.")
    print(f"\nSaved to {MODELS} and {METRICS}")


if __name__ == "__main__":
    main()