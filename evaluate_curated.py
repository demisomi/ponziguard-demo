"""
PonziGuard — Curated Test Set Evaluation
Runs the fine-tuned Stage 2 model against the hand-crafted multilingual
Ponzi test set (curated_ponzi_test.csv) to measure domain-specific
generalisation across English, Pidgin, Hindi, and Indonesian.
"""

import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)

print("=" * 60)
print("  PonziGuard — Curated Test Set Evaluation")
print("=" * 60)

# ── Load curated test set ────────────────────────────────────────────
print("\nLoading curated test set...")
df = pd.read_csv("curated_ponzi_test.csv")
print(f"Total examples: {len(df)}")
print(f"\nLanguage distribution:")
print(df["language"].value_counts())
print(f"\nLabel distribution:")
print(df["label"].value_counts())

# ── Load fine-tuned model ────────────────────────────────────────────
print(f"\nLoading fine-tuned Stage 2 model from ./ponzi_model/")
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "text-classification",
    model="./ponzi_model",
    device=device,
    truncation=True,
    max_length=64
)
print(f"Model loaded on {'GPU' if device == 0 else 'CPU'}")

# ── Run inference ────────────────────────────────────────────────────
print(f"\nClassifying {len(df)} examples...")
predictions = []
scores = []

for i, row in df.iterrows():
    output = classifier(str(row["text"]))[0]
    pred_label = output["label"]
    pred_score = output["score"]

    # Map LABEL_0/LABEL_1 to actual labels if needed
    if pred_label == "LABEL_0":
        pred_label = "legitimate"
    elif pred_label == "LABEL_1":
        pred_label = "ponzi_related"

    predictions.append(pred_label)
    scores.append(pred_score)

df["stage2_pred"]  = predictions
df["stage2_score"] = scores
df["correct"]      = df["stage2_pred"] == df["label"]

# ── Overall metrics ───────────────────────────────────────────────────
y_true = (df["label"] == "ponzi_related").astype(int).values
y_pred = (df["stage2_pred"] == "ponzi_related").astype(int).values

precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
accuracy  = accuracy_score(y_true, y_pred)
cm        = confusion_matrix(y_true, y_pred)

tp = cm[1][1]; fp = cm[0][1]
fn = cm[1][0]; tn = cm[0][0]

print("\n" + "=" * 60)
print("  OVERALL CURATED TEST SET RESULTS")
print("=" * 60)
print(f"  Total examples:     {len(df)}")
print(f"  True Positives:     {tp}")
print(f"  False Positives:    {fp}")
print(f"  False Negatives:    {fn}")
print(f"  True Negatives:     {tn}")
print(f"  Precision:          {precision:.2%}")
print(f"  Recall:             {recall:.2%}")
print(f"  F1 Score:           {f1:.2%}")
print(f"  Accuracy:           {accuracy:.2%}")
print("=" * 60)

# ── Per-language breakdown ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PER-LANGUAGE PERFORMANCE")
print("=" * 60)
print(f"{'Language':<14}{'Count':<8}{'Correct':<10}{'Accuracy':<12}{'F1':<10}")
print("-" * 60)

language_results = []
for lang in df["language"].unique():
    subset = df[df["language"] == lang]
    if len(subset) == 0:
        continue
    y_t = (subset["label"] == "ponzi_related").astype(int).values
    y_p = (subset["stage2_pred"] == "ponzi_related").astype(int).values

    acc  = accuracy_score(y_t, y_p)
    try:
        f1_l = f1_score(y_t, y_p, pos_label=1, zero_division=0)
    except:
        f1_l = 0.0
    correct = int(subset["correct"].sum())

    print(f"{lang:<14}{len(subset):<8}{correct:<10}{acc:.2%}      {f1_l:.2%}")
    language_results.append({
        "language": lang, "count": len(subset),
        "correct": correct, "accuracy": acc, "f1": f1_l
    })

print("=" * 60)

# ── Misclassified examples ────────────────────────────────────────────
wrong = df[~df["correct"]]
if len(wrong) > 0:
    print(f"\n── Misclassified Examples ({len(wrong)}) ──────────────")
    for _, row in wrong.iterrows():
        print(f"\n  Language: {row['language']} | True: {row['label']} | Predicted: {row['stage2_pred']}")
        print(f"  Text: {str(row['text'])[:120]}...")
else:
    print("\n  No misclassifications — 100% accuracy on curated test set.")

# ── Save results ──────────────────────────────────────────────────────
df.to_csv("curated_test_results.csv", index=False)
print(f"\n  Full results saved to: curated_test_results.csv")
print("  This evaluation supplements the primary Stage 2 results.")
