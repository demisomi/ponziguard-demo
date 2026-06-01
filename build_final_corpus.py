"""
PonziGuard - Final Combined Dataset Builder
Merges all data sources into a single expanded corpus that is
substantially more Ponzi-specialised than the original general spam
collection.

Sources merged:
  1. Combined HuggingFace datasets    (telegram_data.csv ~14,609)
  2. LLM-generated synthetic Ponzi    (synthetic_ponzi_messages.csv ~600)
  3. Regulator-mined Ponzi messages   (regulator_ponzi_messages.csv)

Output:
  telegram_data.csv (overwrites previous; final training corpus)

Run order:
  1. python combine_datasets.py            # produces telegram_data.csv
  2. python generate_synthetic_ponzi.py    # produces synthetic_ponzi_messages.csv
  3. python mine_regulator_documents.py    # produces regulator_ponzi_messages.csv
  4. python build_final_corpus.py          # this script
  5. python stage2_finetune.py             # retrain Stage 2 on final corpus
"""

import pandas as pd
import os

print("=" * 60)
print("  PonziGuard - Final Combined Dataset Builder")
print("=" * 60)

all_parts = []

# ─── 1. Load existing combined HuggingFace corpus ────────────────────
print("\n── Loading combined HuggingFace dataset ──")
if os.path.exists("telegram_data.csv"):
    df1 = pd.read_csv("telegram_data.csv")
    df1 = df1[["text", "label"]].dropna()
    df1["origin"] = "huggingface"
    print(f"  Loaded {len(df1):,} messages")
    all_parts.append(df1)
else:
    print("  WARNING: telegram_data.csv not found. Run combine_datasets.py first.")

# ─── 2. Load LLM synthetic Ponzi messages ────────────────────────────
print("\n── Loading LLM-generated synthetic Ponzi data ──")
if os.path.exists("synthetic_ponzi_messages.csv"):
    df2 = pd.read_csv("synthetic_ponzi_messages.csv")
    df2 = df2[["text", "label"]].dropna()
    df2["origin"] = "llm_synthetic"
    print(f"  Loaded {len(df2):,} messages (all ponzi_related)")
    all_parts.append(df2)
else:
    print("  WARNING: synthetic_ponzi_messages.csv not found.")
    print("  Run generate_synthetic_ponzi.py to produce this file.")

# ─── 3. Load regulator-mined messages ────────────────────────────────
print("\n── Loading regulator-mined Ponzi messages ──")
if os.path.exists("regulator_ponzi_messages.csv"):
    df3 = pd.read_csv("regulator_ponzi_messages.csv")
    df3 = df3[["text", "label"]].dropna()
    df3["origin"] = "regulator_mined"
    print(f"  Loaded {len(df3):,} regulator-verified messages")
    all_parts.append(df3)
else:
    print("  WARNING: regulator_ponzi_messages.csv not found.")
    print("  Run mine_regulator_documents.py and review output before merging.")

if not all_parts:
    print("\n  ERROR: No sources loaded. Exiting.")
    exit(1)

# ─── Merge ────────────────────────────────────────────────────────────
print("\n── Merging all sources ──")
combined = pd.concat(all_parts, ignore_index=True)
combined["text"] = combined["text"].astype(str).str.strip()
combined = combined[combined["text"].str.len() > 10]
combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

print(f"  Combined raw:   {len(combined):,} messages")
print(f"  By origin:")
print(combined["origin"].value_counts().to_string())
print(f"  By label:")
print(combined["label"].value_counts().to_string())

# ─── Final balance check ──────────────────────────────────────────────
ponzi = (combined["label"] == "ponzi_related").sum()
legit = (combined["label"] == "legitimate").sum()
ratio = ponzi / (ponzi + legit) * 100
print(f"\n  Ponzi:      {ponzi:,} ({ratio:.1f}%)")
print(f"  Legitimate: {legit:,} ({100-ratio:.1f}%)")

# Save
combined[["text", "label"]].to_csv("telegram_data.csv", index=False)
print(f"\n  Saved final corpus to: telegram_data.csv ({len(combined):,} messages)")
print("  Next step: re-run stage2_finetune.py to retrain on the specialised corpus.")
print("=" * 60)
