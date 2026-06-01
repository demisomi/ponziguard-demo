"""
PonziGuard - Latency Benchmark
Measures inference latency per stage and end-to-end, to address the
sub-100 millisecond target specified in Chapter 3.

Runs each stage on a sample of messages and reports:
  - Mean latency per message (ms)
  - Median latency per message (ms)
  - p95 latency per message (ms)
  - p99 latency per message (ms)

Output: latency_benchmark.csv
"""

import pandas as pd
import time
import numpy as np
import torch
from transformers import pipeline as hf_pipeline
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  PonziGuard - Latency Benchmark")
print("=" * 60)

# Load data
print("\nLoading test messages...")
df = pd.read_csv("telegram_data.csv")
sample = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)
print(f"Benchmarking on {len(sample)} messages")

# Stage 1: Keyword filter
RED_FLAGS = [
    "invest", "investment", "profit", "return", "earn", "income",
    "money", "cash", "guaranteed", "double", "triple", "100%", "200%",
    "300%", "500%", "daily profit", "weekly profit", "ponzi", "pyramid",
    "scheme", "referral", "recruit", "downline", "upline", "commission",
    "withdraw", "deposit", "vip", "premium", "exclusive", "limited",
    "act now", "join now", "hurry", "risk free", "no risk", "forex",
    "crypto", "bitcoin", "btc", "trading", "signal", "passive income",
]

def stage1_keyword_filter(text):
    text_lower = str(text).lower()
    flags = [kw for kw in RED_FLAGS if kw in text_lower]
    return len(flags) > 0

# Stage 2: Fine-tuned DistilBERT
print("\nLoading Stage 2 fine-tuned model...")
try:
    device = 0 if torch.cuda.is_available() else -1
    stage2 = hf_pipeline(
        "text-classification",
        model="./ponzi_model",
        device=device,
        truncation=True,
        max_length=64,
    )
    print(f"  Model loaded on {'GPU' if device == 0 else 'CPU'}")
except Exception as e:
    print(f"  ERROR: Could not load model. {e}")
    print("  Run stage2_finetune.py first to produce ./ponzi_model/")
    exit(1)

# Stage 2.5: Investment sub-filter
INVESTMENT_VOCAB = [
    "invest", "investment", "investing", "return", "returns", "roi",
    "profit", "earnings", "income", "trading", "forex", "crypto",
    "bitcoin", "deposit", "withdraw", "package", "tier", "membership",
    "daily profit", "guaranteed return", "ponzi", "pyramid", "mlm",
]

def stage25_subfilter(text):
    text_lower = str(text).lower()
    return any(v in text_lower for v in INVESTMENT_VOCAB)

# Stage 3: Rule reasoner
RULES = [
    ("Guaranteed Returns", ["guaranteed return", "guaranteed profit", "100% guaranteed", "risk-free investment"], 5),
    ("Unrealistic ROI", ["200%", "300%", "double your money", "triple your money"], 5),
    ("Daily Profit", ["daily profit", "daily income", "weekly profit", "earn daily"], 5),
    ("Pyramid", ["pyramid", "ponzi", "mlm", "downline", "upline"], 5),
    ("Known Schemes", ["mmm", "cbex", "racksterli", "mba forex"], 5),
    ("Recruitment", ["referral commission", "refer and earn"], 5),
    ("Tiered Packages", ["vip package", "premium package", "gold package"], 4),
    ("Passive Income", ["passive income", "financial freedom"], 4),
    ("Urgency", ["limited slots", "act now", "join now"], 2),
    ("Unregulated Trading", ["trading bot", "ai trading"], 2),
    ("Withdrawal", ["withdraw", "deposit"], 1),
    ("Social Proof", ["proof of payment", "100% legit"], 1),
]

def stage3_reasoner(text):
    text_lower = str(text).lower()
    total = 0
    for category, patterns, weight in RULES:
        if any(p in text_lower for p in patterns):
            total += weight
    return total

# Benchmark function
def benchmark(name, fn, inputs):
    latencies = []
    for inp in inputs:
        start = time.perf_counter()
        fn(inp)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    return {
        "stage":  name,
        "n":      len(latencies),
        "mean":   np.mean(latencies),
        "median": np.median(latencies),
        "p95":    np.percentile(latencies, 95),
        "p99":    np.percentile(latencies, 99),
        "min":    np.min(latencies),
        "max":    np.max(latencies),
    }

print("\n" + "=" * 60)
print("  BENCHMARKING EACH STAGE")
print("=" * 60)

texts = sample["text"].astype(str).tolist()

print("\nWarming up Stage 2 model...")
for _ in range(3):
    stage2(texts[0])

results = []

print("\nBenchmarking Stage 1 (keyword filter)...")
r1 = benchmark("Stage 1 - Keyword Filter", stage1_keyword_filter, texts)
results.append(r1)

print("Benchmarking Stage 2 (DistilBERT)...")
r2 = benchmark("Stage 2 - DistilBERT", lambda t: stage2(t)[0], texts)
results.append(r2)

print("Benchmarking Stage 2.5 (sub-filter)...")
r25 = benchmark("Stage 2.5 - Sub-Filter", stage25_subfilter, texts)
results.append(r25)

print("Benchmarking Stage 3 (rule reasoner)...")
r3 = benchmark("Stage 3 - Rule Reasoner", stage3_reasoner, texts)
results.append(r3)

def end_to_end(text):
    if not stage1_keyword_filter(text):
        return "dismissed_stage1"
    s2_pred = stage2(text)[0]
    if "ponzi" not in s2_pred["label"].lower() and s2_pred["label"] != "LABEL_1":
        return "dismissed_stage2"
    if not stage25_subfilter(text):
        return "dismissed_stage25"
    return stage3_reasoner(text)

print("Benchmarking End-to-End cascade...")
r_e2e = benchmark("End-to-End Cascade", end_to_end, texts)
results.append(r_e2e)

print("\n" + "=" * 60)
print("  LATENCY RESULTS (milliseconds)")
print("=" * 60)
print(f"{'Stage':<32}{'Mean':<10}{'Median':<10}{'p95':<10}{'p99':<10}")
print("-" * 72)
for r in results:
    print(f"{r['stage']:<32}{r['mean']:<10.2f}{r['median']:<10.2f}{r['p95']:<10.2f}{r['p99']:<10.2f}")

target_ms = 100
print("\n" + "=" * 60)
print(f"  CHAPTER 3 TARGET: < {target_ms} ms end-to-end")
print("=" * 60)
e2e_mean = results[-1]["mean"]
e2e_p95  = results[-1]["p95"]
print(f"  End-to-end mean:   {e2e_mean:.2f} ms  {'PASS' if e2e_mean < target_ms else 'EXCEEDS TARGET'}")
print(f"  End-to-end p95:    {e2e_p95:.2f} ms  {'PASS' if e2e_p95 < target_ms else 'EXCEEDS TARGET'}")

pd.DataFrame(results).to_csv("latency_benchmark.csv", index=False)
print(f"\n  Results saved to: latency_benchmark.csv")
print("=" * 60)
