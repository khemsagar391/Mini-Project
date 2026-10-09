"""SHAP explanations for Stage 1 (Random Forest) and Stage 2 (Isolation Forest).

Explains a small set of saved test flows and writes ml/reports/explanations.csv.

SHAP values show which features pushed a score up or down for one flow.
They describe what the model relies on. They do not prove what caused an attack.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from feature_schema import FEATURE_NAMES

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
REPORTS = HERE / "reports"
N_PER_GROUP = 3  # flows to explain per group
TOP_K = 3        # features listed per explanation


def rf_shap(model, X):
    """SHAP values for a multiclass Random Forest.

    Returns (values, expected): values has shape (n_flows, n_features, n_classes),
    expected has shape (n_classes,). Both are in probability units.
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    if isinstance(raw, list):           # older shap: one array per class
        values = np.stack(raw, axis=-1)
    else:
        values = np.asarray(raw)
    expected = np.asarray(explainer.expected_value, dtype=float)
    n_classes = len(model.classes_)
    if values.shape != (X.shape[0], X.shape[1], n_classes):
        raise RuntimeError(f"unexpected SHAP shape {values.shape}")
    return values, expected


def rf_additivity_error(model, X, values, expected):
    """Largest gap between (expected + sum of SHAP) and the model's probabilities."""
    reconstructed = values.sum(axis=1) + expected
    return float(np.max(np.abs(reconstructed - model.predict_proba(X))))


def if_shap(iforest, X_explain, X_background):
    """SHAP values for Isolation Forest scores (contributions to score_samples).

    Tries the exact TreeExplainer first. Falls back to KernelExplainer, which
    needs a background sample, if the installed shap version does not support
    Isolation Forest. Returns (values, expected, method_name).
    """
    try:
        explainer = shap.TreeExplainer(iforest)
        values = np.asarray(explainer.shap_values(X_explain))
        expected = float(np.asarray(explainer.expected_value).ravel()[0])
        return values, expected, "TreeExplainer (exact)"
    except Exception as exc:
        explainer = shap.KernelExplainer(iforest.score_samples, X_background)
        values = np.asarray(explainer.shap_values(X_explain, nsamples=200))
        expected = float(np.asarray(explainer.expected_value).ravel()[0])
        return values, expected, f"KernelExplainer (approximate; TreeExplainer failed: {type(exc).__name__})"


def top_features(row_values, X_row, k=TOP_K):
    order = np.argsort(-np.abs(row_values))[:k]
    return [(FEATURE_NAMES[j], float(X_row[j]), float(row_values[j])) for j in order]


def describe(top, lead):
    parts = [f"{name} = {val:,.4g} ({impact:+.3f})" for name, val, impact in top]
    return f"{lead} mainly due to: " + "; ".join(parts)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: pick a few flows first, then explain only those ----
    rf_bundle = joblib.load(MODELS / "random_forest_v1.joblib")
    rf, classes = rf_bundle["model"], rf_bundle["label_classes"]
    saved = np.load(MODELS / "random_forest_v1_test_predictions.npz")
    X_all, pred_all, proba_all = saved["X"], saved["pred"].astype(int), saved["proba"]

    benign_code = classes.index("BENIGN")
    attack_idx = np.where(pred_all != benign_code)[0][:N_PER_GROUP]
    normal_idx = np.where(pred_all == benign_code)[0][:N_PER_GROUP]
    pick = np.concatenate([attack_idx, normal_idx])

    X_pick = X_all[pick]
    pred_pick = pred_all[pick]
    proba_pick = proba_all[pick]

    values, expected = rf_shap(rf, X_pick)
    err = rf_additivity_error(rf, X_pick, values, expected)
    print(f"Random Forest: additivity check, largest gap = {err:.2e} "
          f"({'OK' if err < 1e-4 else 'FAILED'})")

    records = []
    for k, flow in enumerate(pick):
        c = pred_pick[k]
        top = top_features(values[k, :, c], X_pick[k])
        lead = f"Predicted {classes[c]}" if c != benign_code else "Predicted normal"
        records.append({
            "stage": "1 Random Forest",
            "flow": int(flow),
            "prediction": classes[c],
            "class_probability": round(float(proba_pick[k, c]), 4),
            "explanation": describe(top, lead),
        })

    # ---- Stage 2: Isolation Forest on the same flows ----
    if_bundle = joblib.load(MODELS / "isolation_forest_v1.joblib")
    iforest, threshold = if_bundle["model"], float(if_bundle["threshold"])

    # Background for the fallback: a fixed spread of the saved test rows, never more than 200.
    background = X_all[:: max(1, len(X_all) // 200)][:200]
    if_values, if_expected, if_method = if_shap(iforest, X_pick, background)
    print(f"Isolation Forest explanations via: {if_method}")

    flags = -iforest.score_samples(X_pick) > threshold
    for k, flow in enumerate(pick):
        vals = -if_values[k]  # SHAP explains score_samples; negate so + means more unusual
        top = top_features(vals, X_pick[k])
        lead = "Flagged unusual" if flags[k] else "Not flagged (normal)"
        records.append({
            "stage": "2 Isolation Forest",
            "flow": int(flow),
            "prediction": "unusual" if flags[k] else "normal",
            "class_probability": None,
            "explanation": describe(top, lead),
        })

    out = pd.DataFrame(records)
    out.to_csv(REPORTS / "explanations.csv", index=False)
    print("\n" + "\n\n".join(
        f"[{r['stage']}] flow {r['flow']}: {r['explanation']}" for r in records))
    print(f"\nSaved to {REPORTS / 'explanations.csv'}")


if __name__ == "__main__":
    main()