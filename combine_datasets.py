"""
PonziGuard - Combined Multi-Source Dataset Builder
Merges multiple thematically aligned fraud/spam datasets into a single
expanded corpus targeting ~15,000-20,000 messages relevant to Ponzi
scheme detection.

Sources:
  1. thehamkercat/telegram-spam-ham    (your original base, ~8.5k)
  2. redasers/difraud (phishing subset) (~15.2k phishing + benign emails)
  3. ucirvine/sms_spam                  (~5.5k SMS spam/ham)
  4. tanquangduong/nigerian-fraud-emails (Nigerian 419 if available)

All sources are remapped to the binary label scheme:
  ponzi_related  = the deceptive/fraud/scam class
  legitimate     = the benign/ham class

A financial content filter is then applied to keep only messages
relevant to the Ponzi detection domain.
"""

import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# ── Investment / fraud vocabulary filter ─────────────────────────────
PONZI_KEYWORDS = [
    "invest", "profit", "returns", "roi", "referral", "daily", "withdrawal",
    "deposit", "earn", "bonus", "guaranteed", "risk free", "passive income",
    "trading", "crypto", "forex", "scheme", "join", "register", "percentage",
    "percent", "%", "double", "triple", "money", "cash", "fund", "capital",
    "payout", "withdraw", "interest", "dividend", "yield", "bank", "transfer",
    "account", "naira", "dollar", "loan", "wire", "western union", "moneygram",
    "fund transfer", "fee", "commission", "winning", "winner", "lottery",
    "inheritance", "claim", "beneficiary", "urgent", "confidential",
    "pyramid", "ponzi", "mlm", "downline", "upline", "membership", "package",
]

def has_financial_content(text):
    t = str(text).lower()
    return any(kw in t for kw in PONZI_KEYWORDS)

all_dataframes = []

# ═══════════════════════════════════════════════════════════════════════
# SOURCE 1: telegram-spam-ham (your existing base)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Source 1: thehamkercat/telegram-spam-ham")
print("=" * 65)
try:
    ds = load_dataset("thehamkercat/telegram-spam-ham")
    df1 = pd.DataFrame(ds["train"])
    df1 = df1.rename(columns={"text_type": "label"})
    df1["label"] = df1["label"].map({"spam": "ponzi_related", "ham": "legitimate"})
    df1["source"] = "telegram-spam-ham"
    df1 = df1[["text", "label", "source"]].dropna()
    print(f"  Loaded {len(df1):,} messages")
    all_dataframes.append(df1)
except Exception as e:
    print(f"  Failed: {e}")

# ═══════════════════════════════════════════════════════════════════════
# SOURCE 2: DIFrauD - phishing subset (15.2k high-quality fraud emails)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Source 2: redasers/difraud (phishing domain)")
print("=" * 65)
try:
    ds = load_dataset("redasers/difraud", "phishing")
    parts = []
    for split in ["train", "validation", "test"]:
        if split in ds:
            parts.append(pd.DataFrame(ds[split]))
    df2 = pd.concat(parts, ignore_index=True)
    # 1 = deceptive/phishing, 0 = benign
    df2["label"] = df2["label"].map({1: "ponzi_related", 0: "legitimate"})
    df2["source"] = "difraud-phishing"
    df2 = df2[["text", "label", "source"]].dropna()
    print(f"  Loaded {len(df2):,} messages")
    all_dataframes.append(df2)
except Exception as e:
    print(f"  Failed: {e}")
    # Try loading without config name
    try:
        ds = load_dataset("redasers/difraud")
        df2 = pd.DataFrame(ds["train"]) if "train" in ds else pd.DataFrame(ds[list(ds.keys())[0]])
        if "label" in df2.columns and "text" in df2.columns:
            df2["label"] = df2["label"].map({1: "ponzi_related", 0: "legitimate"})
            df2["source"] = "difraud"
            df2 = df2[["text", "label", "source"]].dropna()
            print(f"  Loaded {len(df2):,} messages (fallback)")
            all_dataframes.append(df2)
    except Exception as e2:
        print(f"  Fallback also failed: {e2}")

# ═══════════════════════════════════════════════════════════════════════
# SOURCE 3: SMS Spam Collection (5.5k SMS)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Source 3: ucirvine/sms_spam")
print("=" * 65)
try:
    ds = load_dataset("ucirvine/sms_spam")
    df3 = pd.DataFrame(ds["train"])
    df3 = df3.rename(columns={"sms": "text"})
    df3["label"] = df3["label"].map({1: "ponzi_related", 0: "legitimate"})
    df3["source"] = "sms-spam"
    df3 = df3[["text", "label", "source"]].dropna()
    print(f"  Loaded {len(df3):,} messages")
    all_dataframes.append(df3)
except Exception as e:
    print(f"  Failed: {e}")

# ═══════════════════════════════════════════════════════════════════════
# SOURCE 4 (optional): Phishing email dataset
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Source 4: zefang-liu/phishing-email-dataset")
print("=" * 65)
try:
    ds = load_dataset("zefang-liu/phishing-email-dataset")
    df4 = pd.DataFrame(ds["train"])
    # Common column patterns: 'Email Text', 'Email Type'
    if "Email Text" in df4.columns and "Email Type" in df4.columns:
        df4 = df4.rename(columns={"Email Text": "text", "Email Type": "label"})
        df4["label"] = df4["label"].apply(
            lambda x: "ponzi_related" if "phishing" in str(x).lower() else "legitimate"
        )
    df4["source"] = "phishing-email"
    df4 = df4[["text", "label", "source"]].dropna()
    print(f"  Loaded {len(df4):,} messages")
    all_dataframes.append(df4)
except Exception as e:
    print(f"  Failed: {e}")

# ═══════════════════════════════════════════════════════════════════════
# Merge all sources
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MERGING SOURCES")
print("=" * 65)

if not all_dataframes:
    print("  ERROR: No datasets loaded successfully. Exiting.")
    exit(1)

combined = pd.concat(all_dataframes, ignore_index=True)
combined["text"] = combined["text"].astype(str).str.strip()
combined = combined[combined["text"].str.len() > 10]  # remove tiny messages
combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
combined = combined.dropna(subset=["label"])

print(f"\n  Combined raw:           {len(combined):,} messages")
print(f"  Source breakdown:")
print(combined["source"].value_counts().to_string())

# ── Apply financial content filter ───────────────────────────────────
print("\n" + "=" * 65)
print("  APPLYING FINANCIAL CONTENT FILTER")
print("=" * 65)
combined["has_financial_content"] = combined["text"].apply(has_financial_content)
filtered = combined[combined["has_financial_content"]].copy().reset_index(drop=True)

print(f"\n  Before filter:          {len(combined):,}")
print(f"  After financial filter: {len(filtered):,}")
print(f"  Retained:               {len(filtered)/len(combined)*100:.1f}%")

# ── Cap each source so no single dataset dominates ────────────────────
# Optional: balance by capping any source above 8000 messages
MAX_PER_SOURCE = 8000
balanced_parts = []
for src, group in filtered.groupby("source"):
    if len(group) > MAX_PER_SOURCE:
        group = group.sample(n=MAX_PER_SOURCE, random_state=42)
    balanced_parts.append(group)
filtered = pd.concat(balanced_parts, ignore_index=True)

# Also cap legitimate to ~1.5x ponzi to keep training balance reasonable
ponzi_count = (filtered["label"] == "ponzi_related").sum()
max_legit = int(ponzi_count * 1.5)
legit_subset = filtered[filtered["label"] == "legitimate"]
if len(legit_subset) > max_legit:
    legit_subset = legit_subset.sample(n=max_legit, random_state=42)
ponzi_subset = filtered[filtered["label"] == "ponzi_related"]
filtered = pd.concat([ponzi_subset, legit_subset], ignore_index=True)

# Add platform field for compatibility with existing app.py
filtered["platform"] = "Combined Multi-Source"

# Final shuffle
filtered = filtered.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Save ─────────────────────────────────────────────────────────────
filtered.to_csv("telegram_data.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL DATASET SUMMARY")
print("=" * 65)
print(f"  Total messages:                  {len(filtered):,}")
print(f"  Ponzi-related (labelled):        {(filtered['label'] == 'ponzi_related').sum():,}")
print(f"  Legitimate (labelled):           {(filtered['label'] == 'legitimate').sum():,}")
print(f"\n  Source breakdown after filter:")
print(filtered.groupby(["source", "label"]).size().to_string())
print(f"\n  Saved to telegram_data.csv")
print(f"\n  Next step: re-run stage2_finetune.py to retrain with the larger corpus.")
print("=" * 65)
