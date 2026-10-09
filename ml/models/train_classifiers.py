"""Stage 1: train and compare six supervised classifiers on CICIDS2017.

Model selection uses validation data only. Test metrics are reported for
every model but are not used to choose the model.

Hardware:
  - XGBoost runs on the GPU (device="cuda") when XGBoost was built with CUDA
    and a test fit succeeds. Otherwise it falls back to CPU and says so.
  - Logistic Regression, Decision Tree, Random Forest, KNN and SVM are
    scikit-learn estimators and run on the CPU.

Timing: each model records fit time, validation prediction time and test
prediction time, in seconds, in the comparison CSV.

Writes ml/metrics/stage1_comparison.csv and ml/metrics/stage1_per_class_test.csv.
Does not save model files (that is Phase 6).
"""
import glob
import time
from pathlib import Path
from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, f1_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from data_preprocessing import load_cicids_file, time_split
from feature_schema import FEATURE_NAMES, compute_features

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw" / "cicids2017"
METRICS = HERE / "metrics"
SEED = 42
EVAL_CAP = 20_000       # max rows per class for validation and test scoring
SLOW_TRAIN_CAP = 1_500  # max rows per class for training KNN and SVM


def xgb_device() -> str:
    """Return 'cuda' if XGBoost can train on the GPU, otherwise 'cpu'."""
    try:
        probe = XGBClassifier(n_estimators=2, device="cuda", tree_method="hist")
        probe.fit(np.random.default_rng(0).random((50, 6)), np.array([0, 1] * 25))
        return "cuda"
    except Exception as exc:
        print(f"  GPU not usable for XGBoost, using CPU. Reason: {exc}")
        return "cpu"


def load_all() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "*" / "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found under {RAW}")
    df = pd.concat([load_cicids_file(Path(f)) for f in files], ignore_index=True)
    return time_split(df)


def cap_per_class(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Random subsample of at most n rows per label, reproducible by seed."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in df.groupby("label").indices.items():
        keep.append(idx if len(idx) <= n else rng.choice(idx, n, replace=False))
    return df.iloc[np.sort(np.concatenate(keep))]


def to_xy(df: pd.DataFrame, enc: LabelEncoder):
    fe = compute_features(
        (df["total_packets_fwd"] + df["total_packets_bwd"]).to_numpy(),
        (df["total_bytes_fwd"] + df["total_bytes_bwd"]).to_numpy(),
        df["duration_us"].to_numpy(),
    )
    X = fe[FEATURE_NAMES].to_numpy(dtype=float)
    y = enc.transform(df["label"]).astype(int)
    return X, y


def scores_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)


def summarise(y, pred, scores, K, benign_code):
    Y = label_binarize(y, classes=list(range(K)))
    is_benign = y == benign_code
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "accuracy": accuracy_score(y, pred),
        "pr_auc_macro": average_precision_score(Y, scores, average="macro"),
        "benign_fpr": float((pred[is_benign] != benign_code).mean()),
    }


def build_models(device: str):
    """(model, uses_capped_training). XGBoost is the only one placed on the GPU."""
    return {
        "Logistic Regression": (make_pipeline(StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced")), False),
        "Decision Tree": (DecisionTreeClassifier(max_depth=20, min_samples_leaf=20,
            class_weight="balanced", random_state=SEED), False),
        "Random Forest": (RandomForestClassifier(n_estimators=100, max_depth=20,
            min_samples_leaf=20, max_samples=0.3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED), False),
        "KNN": (make_pipeline(StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1)), True),
        "SVM": (make_pipeline(StandardScaler(),
            SVC(kernel="rbf", C=1.0, class_weight="balanced")), True),
        "XGBoost": (XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
            tree_method="hist", device=device, objective="multi:softprob",
            random_state=SEED), False),
                    "LightGBM": (LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.1,
            class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1), False),
    }


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()

    device = xgb_device()
    print(f"XGBoost device: {device}")

    t0 = time.perf_counter()
    print("Loading all CICIDS2017 files ...", flush=True)
    df = load_all()
    print(f"  {len(df):,} rows loaded in {time.perf_counter() - t0:.1f} s")

    enc = LabelEncoder().fit(df["label"])
    K = len(enc.classes_)
    benign_code = int(enc.transform(["BENIGN"])[0])

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    X_tr, y_tr = to_xy(train, enc)
    small = cap_per_class(train, SLOW_TRAIN_CAP, SEED)
    X_small, y_small = to_xy(small, enc)

    val_eval = cap_per_class(val, EVAL_CAP, SEED)
    test_eval = cap_per_class(test, EVAL_CAP, SEED)
    X_val, y_val = to_xy(val_eval, enc)
    X_test, y_test = to_xy(test_eval, enc)
    print(f"  train full {len(y_tr):,} | train capped {len(y_small):,} | "
          f"val eval {len(y_val):,} | test eval {len(y_test):,}")

    rows, fitted = [], {}
    for name, (model, use_capped) in build_models(device).items():
        Xtr, ytr = (X_small, y_small) if use_capped else (X_tr, y_tr)
        print(f"\nTraining {name} on {len(ytr):,} rows ...", flush=True)
        model_start = time.perf_counter()

        t0 = time.perf_counter()
        if name == "XGBoost":
            model.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        else:
            model.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0

        row = {"model": name, "train_rows": len(ytr), "fit_seconds": round(fit_s, 1)}
        for split_name, (X, y) in {"val": (X_val, y_val), "test": (X_test, y_test)}.items():
            t0 = time.perf_counter()
            pred = model.predict(X).astype(int)
            scores = scores_of(model, X)
            pred_s = time.perf_counter() - t0
            for k, v in summarise(y, pred, scores, K, benign_code).items():
                row[f"{split_name}_{k}"] = round(float(v), 4)
            row[f"{split_name}_predict_seconds"] = round(pred_s, 2)

        row["total_seconds"] = round(time.perf_counter() - model_start, 1)
        row["device"] = device if name == "XGBoost" else "cpu"
        rows.append(row)
        fitted[name] = model
        print(f"  fit {row['fit_seconds']} s | total {row['total_seconds']} s | "
              f"val macro-F1 {row['val_macro_f1']:.4f} | "
              f"val PR-AUC {row['val_pr_auc_macro']:.4f} | "
              f"BENIGN FPR {row['val_benign_fpr']:.4f}")

    result = pd.DataFrame(rows)
    result.to_csv(METRICS / "stage1_comparison.csv", index=False)

    print("\nComparison (selection uses VALIDATION columns only):")
    cols = ["model", "device", "train_rows", "fit_seconds", "val_predict_seconds",
            "total_seconds", "val_macro_f1", "val_pr_auc_macro", "val_accuracy",
            "val_benign_fpr"]
    print(result[cols].sort_values("val_macro_f1", ascending=False).to_string(index=False))

    best = result.sort_values("val_macro_f1", ascending=False).iloc[0]["model"]
    print(f"\nHighest validation macro-F1: {best}")
    print("Test metrics are for reference only; the decision is yours.")
    print("\nTest columns, all models:")
    print(result[["model", "test_macro_f1", "test_pr_auc_macro",
                  "test_accuracy", "test_benign_fpr",
                  "test_predict_seconds"]].to_string(index=False))

    model = fitted[best]
    pred = model.predict(X_test).astype(int)
    report = classification_report(y_test, pred, labels=list(range(K)),
                                   target_names=enc.classes_, output_dict=True,
                                   zero_division=0)
    per_class = pd.DataFrame(report).T
    per_class.to_csv(METRICS / "stage1_per_class_test.csv")
    print(f"\nPer-class test results for {best}:")
    print(per_class.round(4).to_string())

    total = time.perf_counter() - run_start
    print(f"\nTotal run time: {total / 60:.1f} minutes")
    print(f"Saved to {METRICS}")


if __name__ == "__main__":
    main()