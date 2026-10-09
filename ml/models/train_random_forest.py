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
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from data_preprocessing import TRAIN_END, VAL_END  # noqa: F401 (documents the split)
from feature_schema import FEATURE_NAMES, SCHEMA_VERSION
from train_classifiers import (SEED, EVAL_CAP, cap_per_class, load_all,
                               summarise, to_xy)

import sys
from pathlib import Path

MODELS = HERE / "models"
NAME = "random_forest_v1"
PARAMS = dict(n_estimators=100, max_depth=20, min_samples_leaf=20,
              max_samples=0.3, class_weight="balanced_subsample",
              n_jobs=-1, random_state=SEED)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)

    print("Loading CICIDS2017 ...", flush=True)
    df = load_all()
    enc = LabelEncoder().fit(df["label"])
    K = len(enc.classes_)
    benign_code = int(enc.transform(["BENIGN"])[0])

    train = df[df["split"] == "train"]
    X_tr, y_tr = to_xy(train, enc)
    X_val, y_val = to_xy(cap_per_class(df[df["split"] == "val"], EVAL_CAP, SEED), enc)
    test_eval = cap_per_class(df[df["split"] == "test"], EVAL_CAP, SEED)
    X_test, y_test = to_xy(test_eval, enc)
    print(f"Training on {len(y_tr):,} flows", flush=True)

    t0 = time.perf_counter()
    model = RandomForestClassifier(**PARAMS).fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    print(f"Fit took {fit_s:.1f} s", flush=True)

    val_pred = model.predict(X_val).astype(int)
    val_metrics = summarise(y_val, val_pred, model.predict_proba(X_val), K, benign_code)
    test_pred = model.predict(X_test).astype(int)
    test_scores = model.predict_proba(X_test)
    test_metrics = summarise(y_test, test_pred, test_scores, K, benign_code)

    joblib.dump({"model": model, "label_classes": list(enc.classes_)},
                MODELS / f"{NAME}.joblib", compress=3)
    np.savez(MODELS / f"{NAME}_test_predictions.npz",
             X=X_test, pred=test_pred, proba=test_scores)

    meta = {
        "model_name": NAME,
        "algorithm": "sklearn RandomForestClassifier",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "feature_schema_version": SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "label_classes": list(enc.classes_),
        "split": "time-ordered 70/15/15 within each label (not a true session split)",
        "train_flows": int(len(y_tr)),
        "params": PARAMS,
        "fit_seconds": round(fit_s, 1),
        "versions": {"python": "3.12.10", "scikit-learn": sklearn.__version__},
        "validation": {k: round(float(v), 4) for k, v in val_metrics.items()},
        "test_reference_only": {k: round(float(v), 4) for k, v in test_metrics.items()},
    }
    (MODELS / f"{NAME}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nAccuracy (validation): {val_metrics['accuracy']:.4f} "
          f"({val_metrics['accuracy'] * 100:.1f}% of flows labelled correctly)")
    print(f"Attack score (validation): {val_metrics['macro_f1']:.4f} "
          f"(earlier comparison: 0.6711)")
    print(f"False alarm rate (validation): {val_metrics['benign_fpr']:.4f} "
          f"(earlier comparison: 0.1042)")
    print(f"\nSaved to {MODELS}")


if __name__ == "__main__":
    main()