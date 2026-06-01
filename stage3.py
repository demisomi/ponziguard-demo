"""
PonziGuard - Stage 3 v2: Ponzi-Specialised Reasoner
Adds two improvements over the original stage3.py:

  1. STAGE 2.5 (Sub-Filter): An intermediate filter between Stage 2 and
     Stage 3 that distinguishes investment-fraud messages from
     non-investment fraud (phishing, 419, fake invoices, romance scams).
     Only investment-fraud messages proceed to Stage 3 Ponzi scoring.

  2. REWEIGHTED RULES: Stage 3 rules unique to Ponzi schemes
     (guaranteed returns, pyramid structure, recruitment, named schemes,
     tiered packages, daily profit) carry HIGH weights (4-5).
     Rules that appear in general fraud too (urgency, withdrawal, crypto
     generally, social proof) carry LOW weights (1-2).

This shifts Stage 3 from a general fraud detector to a
Ponzi-specialised reasoner.
"""

import pandas as pd
import json
from datetime import datetime

print("=" * 60)
print("  PonziGuard - Stage 3 v2: Ponzi-Specialised Reasoner")
print("=" * 60)

# ─── STAGE 2.5: Investment-Fraud Sub-Filter ───────────────────────────
INVESTMENT_VOCABULARY = [
    "invest", "investment", "investing", "investor", "portfolio",
    "return", "returns", "roi", "yield", "interest rate",
    "profit", "profits", "earnings", "income",
    "passive income", "compound", "capital", "principal",
    "trading", "trader", "trade", "trades", "forex", "fx",
    "crypto", "bitcoin", "btc", "eth", "ethereum", "cryptocurrency",
    "stock", "stocks", "shares", "equities", "bond", "bonds",
    "mutual fund", "etf", "hedge fund", "asset", "assets",
    "deposit", "withdraw", "withdrawal", "payout", "payouts",
    "package", "plan", "tier", "membership", "subscription",
    "compound interest", "compounding",
    "daily profit", "weekly profit", "monthly profit",
    "daily return", "weekly return", "monthly return",
    "daily income", "guaranteed return", "guaranteed profit",
    "guaranteed payout", "double your", "triple your",
    "multiply your", "grow your money", "grow your investment",
    "ponzi", "pyramid", "scheme", "mlm", "matrix",
    "downline", "upline", "referral commission",
    "mmm", "loom", "racksterli", "cbex", "mba forex",
    "twinkas", "imagine global", "chinmark",
    "high return", "high yield", "hyip", "auto trading",
    "trading bot", "ai trading", "algo trading",
    "mining", "staking", "defi", "nft investment",
]

def is_investment_related(text):
    text_lower = str(text).lower()
    return any(vocab in text_lower for vocab in INVESTMENT_VOCABULARY)

# ─── STAGE 3: Reweighted Rules ────────────────────────────────────────
RED_FLAG_RULES = [
    # Weight 5: Ponzi-unique
    {"id": "RF01", "category": "Guaranteed Returns",
     "description": "Promises specific guaranteed return rates (Ponzi signature)",
     "patterns": ["guaranteed return", "guaranteed profit", "guaranteed payout",
                  "guaranteed income", "guaranteed daily", "guaranteed weekly",
                  "100% guaranteed", "100% safe", "100% secure investment",
                  "risk free investment", "risk-free investment",
                  "no risk investment", "zero risk", "assured returns"],
     "weight": 5},
    {"id": "RF02", "category": "Unrealistic ROI",
     "description": "Claims unrealistic return percentages typical of Ponzi schemes",
     "patterns": ["200%", "300%", "400%", "500%", "1000%",
                  "double your money", "triple your money",
                  "multiply your", "10x returns", "100x",
                  "5% daily", "10% daily", "20% daily",
                  "50% return", "100% return"],
     "weight": 5},
    {"id": "RF03", "category": "Daily/Weekly Profit Claims",
     "description": "Promises recurring daily or weekly payouts (Ponzi signature)",
     "patterns": ["daily profit", "daily income", "daily return", "daily payout",
                  "weekly profit", "weekly income", "weekly return", "weekly payout",
                  "monthly profit", "monthly income", "monthly return",
                  "earn daily", "paid daily", "profit daily",
                  "passive daily", "automated daily"],
     "weight": 5},
    {"id": "RF04", "category": "Pyramid / Network Structure",
     "description": "Explicit pyramid, MLM, or network marketing structure",
     "patterns": ["pyramid", "ponzi", "ponzi scheme",
                  "network marketing", "multi-level marketing", "mlm",
                  "matrix", "downline", "upline",
                  "level 1", "level 2", "level 3",
                  "tier 1", "tier 2", "tier 3"],
     "weight": 5},
    {"id": "RF05", "category": "Known Ponzi Scheme References",
     "description": "Direct references to documented Ponzi schemes",
     "patterns": ["mmm", "loom money", "cbex", "racksterli", "racksterly",
                  "mba forex", "imagine global", "chinmark",
                  "motivators", "twinkas", "givers forum",
                  "billion coin", "onecoin", "bitconnect"],
     "weight": 5},
    {"id": "RF06", "category": "Recruitment Incentives",
     "description": "Pays commission for recruiting new investors (Ponzi mechanic)",
     "patterns": ["referral commission", "referral bonus",
                  "refer and earn", "recruit and earn",
                  "downline commission", "matrix commission",
                  "10% commission", "20% commission", "30% commission",
                  "lifetime commission", "recurring commission",
                  "binary plan", "unilevel plan"],
     "weight": 5},
    # Weight 4: Strongly Ponzi
    {"id": "RF07", "category": "Tiered Investment Packages",
     "description": "VIP/Gold/Premium investment tiers (Ponzi structure)",
     "patterns": ["starter package", "bronze package", "silver package",
                  "gold package", "platinum package", "diamond package",
                  "vip package", "premium package", "elite package",
                  "vip member", "elite member", "exclusive member",
                  "upgrade your account", "upgrade plan"],
     "weight": 4},
    {"id": "RF08", "category": "Passive Income Claims",
     "description": "Earnings without work (Ponzi marketing language)",
     "patterns": ["passive income", "passive earnings",
                  "earn while you sleep", "money while you sleep",
                  "no work required", "automated earnings", "set and forget",
                  "financial freedom", "financial independence",
                  "quit your job", "fire your boss", "retire early"],
     "weight": 4},
    # Weight 2: Mixed fraud
    {"id": "RF09", "category": "Urgency / Scarcity",
     "description": "Time pressure tactics (also seen in general fraud)",
     "patterns": ["limited slots", "limited offer", "limited time",
                  "act now", "join now", "don't miss", "do not miss",
                  "hurry", "closing soon", "last chance",
                  "only a few spots", "slots filling fast"],
     "weight": 2},
    {"id": "RF10", "category": "Unregulated Crypto / Forex",
     "description": "Mentions unregulated trading platforms",
     "patterns": ["auto trading bot", "trading bot", "ai trading",
                  "forex signals", "crypto signals", "trading signals",
                  "mining bot", "crypto mining app",
                  "unregulated", "no license required", "no kyc"],
     "weight": 2},
    # Weight 1: Weak signals
    {"id": "RF11", "category": "Withdrawal / Deposit Language",
     "description": "Generic transaction terms (common across all scams)",
     "patterns": ["withdraw", "withdrawal", "deposit",
                  "wallet address", "send to wallet",
                  "transfer to", "pay to", "send funds"],
     "weight": 1},
    {"id": "RF12", "category": "Social Proof Manipulation",
     "description": "Fake testimonials (common across general fraud)",
     "patterns": ["proof of payment", "screenshot", "testimonial",
                  "i just received", "i just withdrew", "it works",
                  "100% legit", "real and legit", "trusted platform",
                  "no scam", "verified"],
     "weight": 1},
]

# Risk thresholds (raised for new weight scale)
RISK_LEVELS = {
    "CRITICAL": (15, "Multiple Ponzi-unique signals confirm scheme promotion"),
    "HIGH":     (10, "Strong combination of Ponzi-specific patterns"),
    "MEDIUM":   (5,  "Some Ponzi indicators present"),
    "LOW":      (1,  "Weak or non-specific fraud signals"),
}

def get_risk_level(score):
    for level, (threshold, desc) in RISK_LEVELS.items():
        if score >= threshold:
            return level, desc
    return "LOW", RISK_LEVELS["LOW"][1]

def analyse_message(text, message_id=None, platform="Telegram"):
    text_lower = str(text).lower()
    triggered = []
    total_weight = 0
    for rule in RED_FLAG_RULES:
        matched = [p for p in rule["patterns"] if p in text_lower]
        if matched:
            triggered.append({
                "rule_id": rule["id"], "category": rule["category"],
                "description": rule["description"], "matched": matched,
                "weight": rule["weight"],
            })
            total_weight += rule["weight"]
    risk_level, risk_description = get_risk_level(total_weight)
    if triggered:
        ponzi_unique = sum(1 for r in triggered if r["weight"] >= 4)
        explanation = (
            f"Triggered {len(triggered)} rule(s) including "
            f"{ponzi_unique} Ponzi-specific pattern(s). "
            f"Risk score: {total_weight}. "
            f"Categories: {', '.join(r['category'] for r in triggered)}."
        )
    else:
        explanation = "No structured red flags detected by rule engine."
    return {
        "alert_id":         f"PG-{datetime.now().strftime('%Y%m%d')}-{str(message_id).zfill(5)}",
        "timestamp":        datetime.now().isoformat(),
        "platform":         platform,
        "text":             text[:300],
        "risk_score":       total_weight,
        "risk_level":       risk_level,
        "risk_description": risk_description,
        "flags_triggered":  len(triggered),
        "ponzi_unique_flags": sum(1 for r in triggered if r["weight"] >= 4),
        "rules":            triggered,
        "explanation":      explanation,
        "stage3_alert":     total_weight >= 5,
    }

# ─── Pipeline execution ──────────────────────────────────────────────
print("\nLoading Stage 2 results...")
try:
    s2_df = pd.read_csv("stage2_finetuned_results.csv")
    confirmed = s2_df[s2_df["stage2_pred_label"] == "ponzi_related"].copy().reset_index(drop=True)
    print(f"Stage 2 confirmed ponzi_related: {len(confirmed):,}")
except FileNotFoundError:
    print("ERROR: stage2_finetuned_results.csv not found. Run stage2_finetune.py first.")
    exit(1)

# Stage 2.5: Apply investment-fraud sub-filter
print(f"\nApplying Stage 2.5 investment-fraud sub-filter...")
confirmed["is_investment_fraud"] = confirmed["text"].apply(is_investment_related)
investment_subset = confirmed[confirmed["is_investment_fraud"]].copy().reset_index(drop=True)
non_investment = confirmed[~confirmed["is_investment_fraud"]].copy().reset_index(drop=True)
print(f"  Stage 2.5 retained (investment-related): {len(investment_subset):,}")
print(f"  Stage 2.5 dismissed (non-investment fraud): {len(non_investment):,}")
if len(non_investment) > 0:
    print(f"  Examples of dismissed messages:")
    for txt in non_investment["text"].head(3):
        print(f"    - {str(txt)[:100]}...")

# Stage 3
print(f"\nRunning Stage 3 v2 reasoner on {len(investment_subset):,} investment-related messages...")
alerts = []
for i, row in investment_subset.iterrows():
    alert = analyse_message(text=row["text"], message_id=i, platform="Telegram")
    alert["true_label"] = row.get("true_label", row.get("label", "unknown"))
    alerts.append(alert)

for i, row in non_investment.iterrows():
    alert = analyse_message(text=row["text"], message_id=f"NI{i}", platform="Telegram")
    alert["stage3_alert"] = False
    alert["risk_level"] = "LOW"
    alert["risk_score"] = 0
    alert["explanation"] = "Dismissed by Stage 2.5: non-investment fraud (likely phishing or 419-style)"
    alert["true_label"] = row.get("true_label", row.get("label", "unknown"))
    alerts.append(alert)

alerts_df = pd.DataFrame(alerts)

# Summary
total          = len(alerts_df)
stage3_alerts  = alerts_df["stage3_alert"].sum()
critical_count = (alerts_df["risk_level"] == "CRITICAL").sum()
high_count     = (alerts_df["risk_level"] == "HIGH").sum()
medium_count   = (alerts_df["risk_level"] == "MEDIUM").sum()
low_count      = (alerts_df["risk_level"] == "LOW").sum()

print("\n" + "=" * 60)
print("  STAGE 3 v2 RESULTS")
print("=" * 60)
print(f"  Stage 2 confirmed:            {total:,}")
print(f"  Stage 2.5 retained:           {len(investment_subset):,}")
print(f"  Stage 2.5 dismissed:          {len(non_investment):,}")
print(f"  Stage 3 alerts (Ponzi):       {stage3_alerts:,}")
print(f"\n  Risk Level Breakdown:")
print(f"    CRITICAL: {critical_count:,}")
print(f"    HIGH:     {high_count:,}")
print(f"    MEDIUM:   {medium_count:,}")
print(f"    LOW:      {low_count:,}")
print("=" * 60)

flat_df = alerts_df[[
    "alert_id", "timestamp", "platform", "text", "true_label",
    "risk_score", "risk_level", "flags_triggered", "ponzi_unique_flags",
    "explanation", "stage3_alert"
]].copy()
flat_df.to_csv("stage3_alerts.csv", index=False)
print(f"\n  Alerts saved to: stage3_alerts.csv")

with open("stage3_alerts.json", "w", encoding="utf-8") as f:
    json.dump(alerts, f, indent=2, ensure_ascii=False)
print(f"  Full alerts saved to: stage3_alerts.json")

print("\n── Top 5 CRITICAL Alerts (Ponzi-specific) ────────────────")
critical = alerts_df[alerts_df["risk_level"] == "CRITICAL"].nlargest(5, "risk_score")
for _, row in critical.iterrows():
    print(f"\n  [{row['alert_id']}] Score: {row['risk_score']} | Ponzi-unique flags: {row['ponzi_unique_flags']}")
    print(f"  Text: {str(row['text'])[:100]}...")
    print(f"  Explanation: {row['explanation']}")