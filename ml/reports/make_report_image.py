
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to file, no window
import matplotlib.pyplot as plt
import pandas as pd

import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]  # the ml/ folder
HERE = ML
sys.path[:0] = [str(ML), str(ML / "models"), str(ML / "reports")]
COMP = HERE / "metrics" / "stage1_comparison.csv"
PER_CLASS = HERE / "metrics" / "stage1_per_class_test.csv"
OUT = HERE / "reports" / "stage1_report.png"


def table_block(ax, title, df, col_formats):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=8)
    cells = []
    for _, row in df.iterrows():
        cells.append([fmt(row[c]) if c in col_formats else str(row[c])
                      for c in df.columns])
    tbl = ax.table(cellText=cells, colLabels=list(df.columns),
                   loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dfe6ee")
            cell.set_text_props(fontweight="bold")


def fmt(v):
    return f"{v:,.3f}" if isinstance(v, float) else f"{v:,}"


def main():
    comp = pd.read_csv(COMP)
    per = pd.read_csv(PER_CLASS, index_col=0)

    val = comp.sort_values("val_macro_f1", ascending=False)[[
        "model", "device", "train_rows", "fit_seconds", "val_predict_seconds",
        "val_macro_f1", "val_pr_auc_macro", "val_accuracy", "val_benign_fpr"]].rename(columns={
        "model": "Model", "device": "Device", "train_rows": "Train flows",
        "fit_seconds": "Fit (s)", "val_predict_seconds": "Predict (s)",
        "val_macro_f1": "Macro-F1", "val_pr_auc_macro": "PR-AUC",
        "val_accuracy": "Accuracy", "val_benign_fpr": "BENIGN FPR"})

    test = comp.sort_values("test_macro_f1", ascending=False)[[
        "model", "test_macro_f1", "test_pr_auc_macro", "test_accuracy",
        "test_benign_fpr"]].rename(columns={
        "model": "Model", "test_macro_f1": "Macro-F1", "test_pr_auc_macro": "PR-AUC",
        "test_accuracy": "Accuracy", "test_benign_fpr": "BENIGN FPR"})

    keep = [c for c in per.index if c not in ("accuracy", "macro avg", "weighted avg")]
    cls = per.loc[keep, ["precision", "recall", "f1-score", "support"]].reset_index()
    cls.columns = ["Class", "Precision", "Recall", "F1", "Test flows"]
    cls["Test flows"] = cls["Test flows"].astype(int)

    fig = plt.figure(figsize=(12, 22), dpi=150)
    fig.suptitle("RouteSense AI: Stage 1 Classifier Comparison (CICIDS2017)",
                 fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.05, 0.955,
             "2,830,743 flows | 70/15/15 time split within each class | "
             "features: packets, bytes, duration, rates, avg packet size",
             fontsize=9, color="#444")

    ax1 = fig.add_axes([0.03, 0.73, 0.94, 0.20])
    table_block(ax1, "1. Validation results (used for model selection)", val,
                {"Fit (s)", "Predict (s)", "Macro-F1", "PR-AUC", "Accuracy", "BENIGN FPR"})

    ax2 = fig.add_axes([0.03, 0.56, 0.94, 0.14])
    table_block(ax2, "2. Test results (reference only)", test,
                {"Macro-F1", "PR-AUC", "Accuracy", "BENIGN FPR"})

    ax3 = fig.add_axes([0.03, 0.04, 0.94, 0.48])
    table_block(ax3, "3. Per-class test results (XGBoost)", cls,
                {"Precision", "Recall", "F1"})

    fig.text(0.03, 0.005,
             "Limitations: not a true session split | SYN/UDP/ICMP floods not labelled "
             "(inside DoS/DDoS) | Heartbleed, Infiltration, SQL injection have 2-6 test flows | "
             "KNN/SVM trained on 16,432 flows | XGBoost on GPU, others on CPU.",
             fontsize=7.5, color="#444", wrap=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()