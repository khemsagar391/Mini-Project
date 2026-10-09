"""Print a plain-language summary of the Stage 1 results. Read-only.

Reads ml/metrics/stage1_comparison.csv and stage1_per_class_test.csv.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
COMP = HERE / "metrics" / "stage1_comparison.csv"
PER_CLASS = HERE / "metrics" / "stage1_per_class_test.csv"


def pct(x):
    return f"{x * 100:.0f}%"


def grade(score):
    if score >= 0.9:
        return "Strong"
    if score >= 0.6:
        return "Fair"
    if score >= 0.3:
        return "Weak"
    return "Very weak"


def main():
    comp = pd.read_csv(COMP)

    print("=" * 60)
    print("MODEL RANKING (validation data, used to choose)")
    print("=" * 60)
    ranked = comp.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    for i, r in ranked.iterrows():
        print(f"{i + 1}. {r['model']:<20} "
              f"attack score {r['val_macro_f1']:.2f} | "
              f"false alarms on normal traffic {pct(r['val_benign_fpr'])} | "
              f"train time {r['fit_seconds']:.0f} s")

    best = ranked.iloc[0]
    print(f"\nBest on validation: {best['model']}")

    print("\n" + "=" * 60)
    print("FOR REFERENCE ONLY: TEST DATA")
    print("=" * 60)
    t = comp.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
    for _, r in t.iterrows():
        print(f"{r['model']:<20} attack score {r['test_macro_f1']:.2f} | "
              f"false alarms {pct(r['test_benign_fpr'])}")

    print("\n" + "=" * 60)
    print(f"ATTACK TYPES (test data, {best['model']} shown)")
    print("=" * 60)
    per = pd.read_csv(PER_CLASS, index_col=0)
    skip = {"accuracy", "macro avg", "weighted avg"}
    for name, r in per.iterrows():
        if name in skip:
            continue
        flows = int(r["support"])
        if flows < 10:
            verdict = "Too few flows to judge"
        else:
            verdict = grade(r["f1-score"])
        print(f"{name:<28} caught {pct(r['recall']):>4} | "
              f"correct when flagged {pct(r['precision']):>4} | "
              f"{flows:>6} flows | {verdict}")

    print("\n" + "=" * 60)
    print("WHAT THIS MEANS")
    print("=" * 60)
    print("- Strong on common attacks (DDoS, PortScan, FTP-Patator).")
    print("- Weak or missed on Bot, web attacks and slowloris.")
    print("- False alarms on normal traffic are too high to block hosts")
    print("  automatically. Alerts must be confirmed first.")


if __name__ == "__main__":
    main()