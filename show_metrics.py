"""Recover Stage 2 metrics from saved results CSV."""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

df = pd.read_csv("stage2_finetuned_results.csv")

y_true = (df["label"] == "ponzi_related").astype(int).values
y_pred = (df["stage2_pred_label"] == "ponzi_related").astype(int).values

precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
accuracy  = accuracy_score(y_true, y_pred)
cm        = confusion_matrix(y_true, y_pred)
tp, fp = cm[1][1], cm[0][1]
fn, tn = cm[1][0], cm[0][0]

print("=" * 60)
print("  STAGE 2 — FULL TEST SET METRICS")
print("=" * 60)
print(f"  Test set size:    {len(df):,}")
print(f"  True Positives:   {tp:,}")
print(f"  False Positives:  {fp:,}")
print(f"  False Negatives:  {fn:,}")
print(f"  True Negatives:   {tn:,}")
print(f"  Precision:        {precision:.2%}")
print(f"  Recall:           {recall:.2%}")
print(f"  F1 Score:         {f1:.2%}")
print(f"  Accuracy:         {accuracy:.2%}")
print("=" * 60)
