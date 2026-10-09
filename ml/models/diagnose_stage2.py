"""Diagnose Stage 2 on VALIDATION data only. Read-only with respect to the test set.

For each attack class, measures how well the 'unusual' score separates it from
normal traffic (ROC-AUC: 0.5 = no separation, 1.0 = perfect). Compares Isolation
Forest with Local Outlier Factor. Choosing between them must use these numbers.
"""
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]  # the ml/ folder
HERE = ML
sys.path[:0] = [str(ML), str(ML / "models"), str(ML / "reports")]

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder

from train_classifiers import SEED, EVAL_CAP, cap_per_class, load_all, to_xy
import sys
from pathlib import Path

METRICS = HERE / "metrics"
MODELS = HERE / "models"
LOF_TRAIN_CAP = 20_000  # LOF scales with training size; use a sample


def main():
    print("Loading CICIDS2017 ...", flush=True)
    df = load_all()
    enc = LabelEncoder().fit(df["label"])
    benign = int(enc.transform(["BENIGN"])[0])

    train = df[df["split"] == "train"]
    normal_train = train[train["label"] == "BENIGN"]
    X_normal, _ = to_xy(normal_train, enc)
    rng = np.random.default_rng(SEED)
    X_lof = X_normal[rng.choice(len(X_normal), min(LOF_TRAIN_CAP, len(X_normal)), replace=False)]

    val_eval = cap_per_class(df[df["split"] == "val"], EVAL_CAP, SEED)
    X_val, y_val = to_xy(val_eval, enc)
    labels = val_eval["label"].to_numpy()

    iforest = joblib.load(MODELS / "isolation_forest_v1.joblib")["model"]
    unusual = {"Isolation Forest": -iforest.score_samples(X_val)}

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(X_lof)
    unusual["Local Outlier Factor"] = -lof.score_samples(X_val)

    rows = []
    normal = labels == "BENIGN"
    for cls in sorted(set(labels) - {"BENIGN"}):
        m = normal | (labels == cls)
        y = (labels[m] == cls).astype(int)
        row = {"class": cls, "val_flows": int((labels == cls).sum())}
        for name, score in unusual.items():
            row[f"AUC {name}"] = round(roc_auc_score(y, score[m]), 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "stage2_diagnosis_validation.csv", index=False)
    print("\nROC-AUC on validation (0.5 = cannot separate from normal, 1.0 = perfect)")
    print(out.to_string(index=False))
    print(f"\nSaved to {METRICS / 'stage2_diagnosis_validation.csv'}")


if __name__ == "__main__":
    main()