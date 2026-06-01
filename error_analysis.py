"""
PonziGuard - Error Analysis
Categorises misclassifications on the curated multilingual test set to
identify systematic failure modes. This converts a weakness (errors)
into demonstrated analytical depth for thesis defense.

Each error is categorised into one of:
  - LEXICAL_DRIFT       : message uses non-standard or rare vocabulary
  - CODE_MIXED          : message switches between languages
  - SHORT_MESSAGE       : message too short for context (<20 words)
  - FINANCIAL_LEGIT     : legitimate financial message misread as fraud
  - CULTURAL_MARKER     : language-specific scheme name not in keywords
  - NUMERIC_HEAVY       : message dominated by digits/symbols
  - WARNING_MISREAD     : regulator warning treated as fraud (irony)
  - OTHER               : unable to categorise

Output: error_analysis.csv + console report
"""

import pandas as pd
import re

print("=" * 60)
print("  PonziGuard - Error Analysis")
print("=" * 60)

# Load results
try:
    df = pd.read_csv("curated_test_results.csv")
except FileNotFoundError:
    print("\nERROR: curated_test_results.csv not found.")
    print("Run evaluate_curated.py first.")
    exit(1)

# Identify errors
df["error"] = df["stage2_pred"] != df["label"]
errors = df[df["error"]].copy().reset_index(drop=True)
print(f"\nTotal test examples: {len(df)}")
print(f"Total misclassifications: {len(errors)}")
print(f"Error rate: {len(errors)/len(df)*100:.1f}%")

# Categorisation
def categorise_error(row):
    text = str(row["text"]).lower()
    label = row["label"]
    pred = row["stage2_pred"]

    # Warning misread (regulator warnings about Ponzi flagged as Ponzi)
    if label == "legitimate" and pred == "ponzi_related":
        if any(w in text for w in ["warn", "advisory", "beware", "caution", "verify", "sebi", "cbn", "ojk", "sec ", "rbi"]):
            return "WARNING_MISREAD"
        if any(w in text for w in ["bank", "branch", "fixed deposit", "rate", "mutual fund", "stock"]):
            return "FINANCIAL_LEGIT"

    # Short message
    word_count = len(text.split())
    if word_count < 15:
        return "SHORT_MESSAGE"

    # Numeric heavy
    digit_count = sum(1 for c in text if c.isdigit())
    if digit_count > len(text) * 0.2:
        return "NUMERIC_HEAVY"

    # Code-mixed (look for both English and another language indicator)
    code_mixed_indicators = ["bhai", "abeg", "wahala", "saya", "bro", "dijamin", "₦", "₹", "rp"]
    indicator_count = sum(1 for ind in code_mixed_indicators if ind in text)
    if indicator_count >= 2:
        return "CODE_MIXED"

    # Cultural marker missed
    if label == "ponzi_related" and pred == "legitimate":
        cultural = ["mmm", "cbex", "racksterli", "loom", "twinkas", "mba forex"]
        if any(c in text for c in cultural):
            return "CULTURAL_MARKER"

    # Lexical drift (uses unusual scheme vocabulary)
    if label == "ponzi_related" and pred == "legitimate":
        return "LEXICAL_DRIFT"

    return "OTHER"

errors["category"] = errors.apply(categorise_error, axis=1)

# Per-category breakdown
print("\n" + "=" * 60)
print("  ERROR CATEGORIES")
print("=" * 60)
cat_counts = errors["category"].value_counts()
for cat, count in cat_counts.items():
    pct = count / len(errors) * 100
    print(f"  {cat:<20} {count:3d}  ({pct:5.1f}%)")

# Per-category language breakdown
print("\n" + "=" * 60)
print("  CATEGORIES BY LANGUAGE")
print("=" * 60)
breakdown = errors.groupby(["language", "category"]).size().reset_index(name="count")
print(breakdown.to_string(index=False))

# Sample errors per category
print("\n" + "=" * 60)
print("  SAMPLE ERRORS PER CATEGORY")
print("=" * 60)
for cat in cat_counts.index:
    samples = errors[errors["category"] == cat].head(2)
    print(f"\n  --- {cat} ---")
    for _, row in samples.iterrows():
        print(f"  [{row['language']}] True: {row['label']} | Predicted: {row['stage2_pred']}")
        print(f"    Text: {str(row['text'])[:150]}...")

# Save
errors.to_csv("error_analysis.csv", index=False)
print(f"\n\n  Detailed error analysis saved to: error_analysis.csv")
print("=" * 60)
