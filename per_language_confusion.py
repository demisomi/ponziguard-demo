"""
PonziGuard - Per-Language Confusion Matrices
Computes detailed per-language confusion matrices on the curated
multilingual test set, breaking aggregate metrics down by language to
support precise defense argumentation about cross-lingual generalisation.

Output: per_language_confusion.csv + console report
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

print("=" * 60)
print("  PonziGuard - Per-Language Confusion Matrices")
print("=" * 60)

# Load curated test results
try:
    df = pd.read_csv("curated_test_results.csv")
    print(f"\nLoaded {len(df)} curated test results.")
except FileNotFoundError:
    print("\nERROR: curated_test_results.csv not found.")
    print("Run evaluate_curated.py first.")
    exit(1)

# Also try to load cascade results if available
cascade_df = None
try:
    cascade_df = pd.read_csv("curated_cascade_results.csv")
    print(f"Loaded {len(cascade_df)} cascade test results.")
except FileNotFoundError:
    print("(Cascade results not found - skipping cascade per-language analysis)")

# Stage 2 alone — per-language confusion
print("\n" + "=" * 60)
print("  STAGE 2 ALONE — PER-LANGUAGE CONFUSION MATRICES")
print("=" * 60)

stage2_summary = []
for lang in df["language"].unique():
    subset = df[df["language"] == lang]
    y_true = (subset["label"] == "ponzi_related").astype(int).values
    y_pred = (subset["stage2_pred"] == "ponzi_related").astype(int).values

    if len(set(y_true)) == 1:
        # Skip if only one class present
        continue

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = cm[0]
    fp, tn = cm[1]
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    print(f"\n  {lang.upper()} (n={len(subset)})")
    print(f"  ┌─────────────────────┬───────────┬────────────┐")
    print(f"  │                     │ Pred Ponzi│ Pred Legit │")
    print(f"  ├─────────────────────┼───────────┼────────────┤")
    print(f"  │ Actually Ponzi      │    {tp:3d}    │    {fn:3d}     │")
    print(f"  │ Actually Legitimate │    {fp:3d}    │    {tn:3d}     │")
    print(f"  └─────────────────────┴───────────┴────────────┘")
    print(f"  Precision: {prec:.2%} | Recall: {rec:.2%} | F1: {f1:.2%}")

    stage2_summary.append({
        "language":  lang,
        "stage":     "Stage 2 alone",
        "n":         len(subset),
        "tp":        int(tp), "fp": int(fp),
        "fn":        int(fn), "tn": int(tn),
        "precision": prec, "recall": rec, "f1": f1,
    })

# Full cascade per-language
cascade_summary = []
if cascade_df is not None:
    print("\n" + "=" * 60)
    print("  FULL CASCADE (Stage 2 + Stage 3) — PER-LANGUAGE")
    print("=" * 60)

    for lang in cascade_df["language"].unique():
        subset = cascade_df[cascade_df["language"] == lang]
        y_true = (subset["label"] == "ponzi_related").astype(int).values
        y_pred = (subset["final_prediction"] == "ponzi_related").astype(int).values

        if len(set(y_true)) == 1:
            continue

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        tp, fn = cm[0]
        fp, tn = cm[1]
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        print(f"\n  {lang.upper()} (n={len(subset)})")
        print(f"  ┌─────────────────────┬───────────┬────────────┐")
        print(f"  │                     │ Pred Ponzi│ Pred Legit │")
        print(f"  ├─────────────────────┼───────────┼────────────┤")
        print(f"  │ Actually Ponzi      │    {tp:3d}    │    {fn:3d}     │")
        print(f"  │ Actually Legitimate │    {fp:3d}    │    {tn:3d}     │")
        print(f"  └─────────────────────┴───────────┴────────────┘")
        print(f"  Precision: {prec:.2%} | Recall: {rec:.2%} | F1: {f1:.2%}")

        cascade_summary.append({
            "language":  lang,
            "stage":     "Cascade",
            "n":         len(subset),
            "tp":        int(tp), "fp": int(fp),
            "fn":        int(fn), "tn": int(tn),
            "precision": prec, "recall": rec, "f1": f1,
        })

# Save
combined = stage2_summary + cascade_summary
pd.DataFrame(combined).to_csv("per_language_confusion.csv", index=False)

print("\n" + "=" * 60)
print(f"  Summary saved to: per_language_confusion.csv")
print("=" * 60)
