"""
PonziGuard - Threshold Sensitivity Analysis
Sweeps the Stage 3 risk score threshold across its full range and
reports how precision, recall, and F1 vary. Empirically demonstrates
the architectural claim that the threshold is operator-configurable.

Generates:
  - threshold_sensitivity.csv with metrics at each threshold
  - Console table summarising the precision-recall trade-off curve
"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

print("=" * 60)
print("  PonziGuard - Threshold Sensitivity Analysis")
print("=" * 60)

# Load cascade results
try:
    df = pd.read_csv("curated_cascade_results.csv")
    print(f"\nLoaded {len(df)} cascade test results.")
except FileNotFoundError:
    print("\nERROR: curated_cascade_results.csv not found.")
    print("Run evaluate_curated_cascade.py first.")
    exit(1)

# Sweep thresholds
print("\nSweeping Stage 3 risk thresholds from 0 to 20...")
results = []
y_true_all = (df["label"] == "ponzi_related").astype(int).values

# We need the stage3_score for each Stage 2-flagged message
if "stage3_score" not in df.columns:
    print("ERROR: cascade results CSV does not include stage3_score column.")
    print("Make sure evaluate_curated_cascade.py saves stage3_score per row.")
    exit(1)

for threshold in range(0, 21):
    df_copy = df.copy()

    # Reconstruct final prediction at this threshold
    def predict(row):
        if pd.isna(row.get("stage3_score", None)):
            return "legitimate"  # not flagged by Stage 2
        if row["stage2_pred"] != "ponzi_related":
            return "legitimate"
        return "ponzi_related" if row["stage3_score"] >= threshold else "legitimate"

    df_copy["final"] = df_copy.apply(predict, axis=1)
    y_pred = (df_copy["final"] == "ponzi_related").astype(int).values

    if y_pred.sum() == 0:
        prec = 0.0
    else:
        prec = precision_score(y_true_all, y_pred, zero_division=0)
    rec = recall_score(y_true_all, y_pred, zero_division=0)
    f1  = f1_score(y_true_all, y_pred, zero_division=0)
    acc = accuracy_score(y_true_all, y_pred)

    results.append({
        "threshold": threshold,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "accuracy":  acc,
        "alerts":    int(y_pred.sum()),
    })

results_df = pd.DataFrame(results)

# Report
print("\n" + "=" * 60)
print("  THRESHOLD SENSITIVITY RESULTS")
print("=" * 60)
print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Accuracy':<12}{'#Alerts':<10}")
print("-" * 72)
for r in results:
    print(f"{r['threshold']:<12}{r['precision']:<12.2%}{r['recall']:<12.2%}{r['f1']:<12.2%}{r['accuracy']:<12.2%}{r['alerts']:<10}")

# Identify operating points
best_f1 = max(results, key=lambda r: r['f1'])
balanced = next((r for r in results if abs(r['precision'] - r['recall']) < 0.1), None)
recall_optimised = max(results, key=lambda r: r['recall'])
precision_optimised = max(
    (r for r in results if r['recall'] > 0.3),
    key=lambda r: r['precision'],
    default=None
)

print("\n" + "=" * 60)
print("  RECOMMENDED OPERATING POINTS")
print("=" * 60)
print(f"\n  Best F1: threshold={best_f1['threshold']}")
print(f"    Precision {best_f1['precision']:.2%} | Recall {best_f1['recall']:.2%} | F1 {best_f1['f1']:.2%}")
if balanced:
    print(f"\n  Balanced (precision~recall): threshold={balanced['threshold']}")
    print(f"    Precision {balanced['precision']:.2%} | Recall {balanced['recall']:.2%} | F1 {balanced['f1']:.2%}")
print(f"\n  Recall-optimised (regulatory reporting): threshold={recall_optimised['threshold']}")
print(f"    Precision {recall_optimised['precision']:.2%} | Recall {recall_optimised['recall']:.2%}")
if precision_optimised:
    print(f"\n  Precision-optimised (consumer alerts): threshold={precision_optimised['threshold']}")
    print(f"    Precision {precision_optimised['precision']:.2%} | Recall {precision_optimised['recall']:.2%}")

# Save
results_df.to_csv("threshold_sensitivity.csv", index=False)
print(f"\n  Saved to: threshold_sensitivity.csv")
print("=" * 60)
