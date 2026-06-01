"""
PonziGuard — Stage 3 Evaluation on Curated Test Set
Runs Stage 3 rule reasoner on messages flagged as ponzi_related by Stage 2
on the curated test set. Demonstrates how Stage 3 refines Stage 2 predictions
by down-scoring legitimate financial content that Stage 2 over-flagged.
"""

import pandas as pd
from datetime import datetime

print("=" * 60)
print("  Stage 3 Cascade Evaluation on Curated Test Set")
print("=" * 60)

# Same rules as stage3.py
RED_FLAG_RULES = [
    {"id": "RF01", "category": "Guaranteed Returns",
     "patterns": ["guaranteed", "guarantee", "100% guaranteed",
                  "risk free", "risk-free", "no risk", "zero risk",
                  "assured returns", "assured profit", "dijamin", "dijamin aman"],
     "weight": 3},
    {"id": "RF02", "category": "Unrealistic ROI",
     "patterns": ["200%", "300%", "400%", "500%", "1000%",
                  "double your money", "triple your money",
                  "multiply your", "100x", "10x returns",
                  "massive returns", "huge returns", "high returns"],
     "weight": 3},
    {"id": "RF03", "category": "Daily/Weekly Profit Claims",
     "patterns": ["daily profit", "daily income", "daily returns",
                  "weekly profit", "weekly income", "weekly returns",
                  "earn daily", "paid daily", "per day", "per week",
                  "every day", "each day", "per hari", "har roz", "har hafte"],
     "weight": 3},
    {"id": "RF04", "category": "Recruitment / Referral Incentives",
     "patterns": ["referral", "refer a friend", "refer and earn",
                  "recruit", "downline", "upline",
                  "commission", "referral bonus", "invite friends",
                  "bring others", "get others to join", "ajak teman"],
     "weight": 3},
    {"id": "RF05", "category": "Urgency / Scarcity Pressure",
     "patterns": ["limited slots", "limited offer", "limited time",
                  "act now", "join now", "don't miss", "do not miss",
                  "hurry", "closing soon", "last chance",
                  "only a few spots", "slots filling fast",
                  "slot terbatas", "jaldi join"],
     "weight": 2},
    {"id": "RF06", "category": "Withdrawal / Deposit Language",
     "patterns": ["withdraw", "withdrawal", "deposit",
                  "send money", "transfer funds", "bank transfer",
                  "pay to", "send to", "setor", "tarik"],
     "weight": 2},
    {"id": "RF07", "category": "Pyramid / Network Structure",
     "patterns": ["pyramid", "ponzi", "scheme",
                  "network marketing", "multi-level", "mlm",
                  "tier", "level up", "rank up",
                  "downline", "upline", "matrix"],
     "weight": 3},
    {"id": "RF08", "category": "Passive Income Claims",
     "patterns": ["passive income", "passive earnings",
                  "earn while you sleep", "ghar baithe",
                  "financial freedom", "financial independence",
                  "quit your job", "retire early",
                  "no work required", "automated earnings"],
     "weight": 2},
    {"id": "RF09", "category": "Membership / Package Tiers",
     "patterns": ["vip", "premium", "gold package", "silver package",
                  "starter package", "pro plan",
                  "upgrade your account", "upgrade membership",
                  "exclusive membership", "elite tier"],
     "weight": 2},
    {"id": "RF10", "category": "Cryptocurrency / Forex Fraud",
     "patterns": ["crypto investment", "bitcoin investment",
                  "forex signals", "trading signals",
                  "crypto returns", "btc profit",
                  "unregulated", "no license required", "mining bot"],
     "weight": 2},
    {"id": "RF11", "category": "Known Ponzi Scheme References",
     "patterns": ["mmm", "loom money", "cbex", "racksterli",
                  "mba forex", "chinmark", "twinkas"],
     "weight": 3},
    {"id": "RF12", "category": "Social Proof Manipulation",
     "patterns": ["i don withdraw", "i don collect", "no be scam",
                  "i swear", "na legit", "saya sudah buktikan",
                  "main khud", "bhai yeh"],
     "weight": 2},
]

def score_message(text):
    text_lower = str(text).lower()
    triggered = []
    total = 0
    for rule in RED_FLAG_RULES:
        matched = [p for p in rule["patterns"] if p in text_lower]
        if matched:
            triggered.append(rule["category"])
            total += rule["weight"]
    return total, triggered

# ── Load curated Stage 2 results ──────────────────────────────────────
df = pd.read_csv("curated_test_results.csv")
print(f"\nLoaded {len(df)} curated examples with Stage 2 predictions.")

# ── Take only Stage 2 flagged messages ────────────────────────────────
stage2_flagged = df[df["stage2_pred"] == "ponzi_related"].copy()
print(f"Stage 2 flagged: {len(stage2_flagged)}")
print(f"  True Ponzi: {(stage2_flagged['label'] == 'ponzi_related').sum()}")
print(f"  Legit:      {(stage2_flagged['label'] == 'legitimate').sum()}")

# ── Apply Stage 3 ─────────────────────────────────────────────────────
print("\nApplying Stage 3 rule reasoner...")
stage2_flagged["stage3_score"]   = stage2_flagged["text"].apply(lambda t: score_message(t)[0])
stage2_flagged["stage3_flags"]   = stage2_flagged["text"].apply(lambda t: ", ".join(score_message(t)[1]) if score_message(t)[1] else "—")
stage2_flagged["stage3_alert"]   = stage2_flagged["stage3_score"] >= 3

# ── Stage 3 metrics on this subset ────────────────────────────────────
y_true = (stage2_flagged["label"] == "ponzi_related").astype(int).values
y_pred = stage2_flagged["stage3_alert"].astype(int).values

tp = int(((y_pred == 1) & (y_true == 1)).sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum())
fn = int(((y_pred == 0) & (y_true == 1)).sum())
tn = int(((y_pred == 0) & (y_true == 0)).sum())
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy  = (tp + tn) / len(stage2_flagged) if len(stage2_flagged) > 0 else 0

print("\n" + "=" * 60)
print("  STAGE 3 REFINEMENT METRICS (on Stage 2-flagged messages)")
print("=" * 60)
print(f"  Messages processed:  {len(stage2_flagged)}")
print(f"  Stage 3 confirmed:   {y_pred.sum()}")
print(f"  Stage 3 dismissed:   {(y_pred == 0).sum()}")
print(f"\n  Precision: {precision:.2%}")
print(f"  Recall:    {recall:.2%}")
print(f"  F1 Score:  {f1:.2%}")
print(f"  Accuracy:  {accuracy:.2%}")
print("=" * 60)

# ── Compare end-to-end pipeline vs Stage 2 alone ──────────────────────
# End-to-end: message is flagged only if Stage 2 flags AND Stage 3 confirms
df["final_prediction"] = "legitimate"
mask = df["stage2_pred"] == "ponzi_related"
df.loc[mask, "final_prediction"] = "legitimate"  # default
# Apply Stage 3 only to Stage 2 flagged
for idx, row in stage2_flagged.iterrows():
    if row["stage3_alert"]:
        df.loc[idx, "final_prediction"] = "ponzi_related"

y_true_all = (df["label"] == "ponzi_related").astype(int).values
y_pred_s2  = (df["stage2_pred"] == "ponzi_related").astype(int).values
y_pred_e2e = (df["final_prediction"] == "ponzi_related").astype(int).values

def metrics(y_t, y_p):
    tp = int(((y_p == 1) & (y_t == 1)).sum())
    fp = int(((y_p == 1) & (y_t == 0)).sum())
    fn = int(((y_p == 0) & (y_t == 1)).sum())
    tn = int(((y_p == 0) & (y_t == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc  = (tp + tn) / len(y_t) if len(y_t) > 0 else 0
    return prec, rec, f1_, acc, tp, fp, fn, tn

s2_m  = metrics(y_true_all, y_pred_s2)
e2e_m = metrics(y_true_all, y_pred_e2e)

print("\n" + "=" * 60)
print("  END-TO-END CASCADE vs STAGE 2 ALONE (Curated Test)")
print("=" * 60)
print(f"{'Metric':<15}{'Stage 2 Alone':<18}{'Stage 2 + Stage 3':<20}{'Change'}")
print("-" * 60)
print(f"{'Precision':<15}{s2_m[0]:.2%}           {e2e_m[0]:.2%}              {'+' if e2e_m[0]>s2_m[0] else ''}{(e2e_m[0]-s2_m[0])*100:+.1f}pp")
print(f"{'Recall':<15}{s2_m[1]:.2%}          {e2e_m[1]:.2%}             {'+' if e2e_m[1]>s2_m[1] else ''}{(e2e_m[1]-s2_m[1])*100:+.1f}pp")
print(f"{'F1 Score':<15}{s2_m[2]:.2%}           {e2e_m[2]:.2%}              {'+' if e2e_m[2]>s2_m[2] else ''}{(e2e_m[2]-s2_m[2])*100:+.1f}pp")
print(f"{'Accuracy':<15}{s2_m[3]:.2%}           {e2e_m[3]:.2%}              {'+' if e2e_m[3]>s2_m[3] else ''}{(e2e_m[3]-s2_m[3])*100:+.1f}pp")
print("=" * 60)

# ── Merge stage3_score back to main df for threshold analysis ────────
df = df.merge(
    stage2_flagged[["text", "stage3_score", "stage3_flags"]],
    on="text", how="left"
)
df["stage3_score"] = df["stage3_score"].fillna(0).astype(int)

# ── Save ──────────────────────────────────────────────────────────────
df.to_csv("curated_cascade_results.csv", index=False)
print(f"\n  Saved to: curated_cascade_results.csv")

# ── Show how Stage 3 dismissed Stage 2 false positives ────────────────
dismissed_legit = stage2_flagged[
    (~stage2_flagged["stage3_alert"]) & (stage2_flagged["label"] == "legitimate")
]
if len(dismissed_legit) > 0:
    print(f"\n── Stage 3 correctly dismissed {len(dismissed_legit)} Stage 2 false positives ──")
    for _, row in dismissed_legit.head(8).iterrows():
        print(f"\n  Language: {row['language']}")
        print(f"  Text: {str(row['text'])[:100]}...")
        print(f"  Stage 3 score: {row['stage3_score']} — below alert threshold")
