# PonziGuard

**A Generalizable, Real-Time Detector of Social-Media Ponzi Schemes**

Final Year Project — Computer Science, Pan-Atlantic University
Author: Oluwademilade Somide
Supervisor: Dr Taiwo Amoo

---

## Overview

PonziGuard is a three-stage cascade detection framework for identifying
Ponzi scheme promotional content on social media across multiple
languages. This repository contains the prototype implementation of the
framework defended in Chapters 1–3 of the project thesis.

The cascade architecture:

1. **Stage 1 — UltraFast Filter**: keyword-density gate (recall-optimised)
2. **Stage 2 — Multilingual Deep Analyzer**: fine-tuned XLM-RoBERTa-base
3. **Stage 2.5 — Investment-Fraud Sub-Filter**: distinguishes investment scams from general fraud
4. **Stage 3 — Structured Red-Flag Reasoner**: 12 weighted rule categories with risk scoring

<img width="1700" height="1800" alt="methodology_flowchart" src="https://github.com/user-attachments/assets/e52239ea-48d6-4c31-8f00-4d6ebf4c6682" />



---

## Quick Start

### Installation

```bash
# Clone the repo and enter the folder
cd PonziGuard

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
# 1. Build the combined dataset (3 HuggingFace sources + synthetic Ponzi data)
python combine_datasets.py
python build_final_corpus.py

# 2. Fine-tune Stage 2 (XLM-RoBERTa-base)
python stage2_finetune.py

# 3. Run Stage 3 rule reasoner
python stage3.py

# 4. Evaluate on the curated multilingual test set
python evaluate_curated.py
python evaluate_curated_cascade.py

# 5. Launch the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Project Structure

```
PonziGuard/
├── app.py                              Streamlit dashboard (5 tabs)
├── combine_datasets.py                 Builds expanded HuggingFace corpus
├── build_final_corpus.py               Merges synthetic + regulator data
├── stage2_finetune.py                  Fine-tunes XLM-RoBERTa-base
├── stage3.py                           Runs rule reasoner v2
├── evaluate_curated.py                 Evaluates on curated multilingual set
├── evaluate_curated_cascade.py         Evaluates full cascade
├── benchmark_latency.py                Measures per-stage inference latency
├── synthetic_ponzi_messages.csv        401 hand-crafted Ponzi messages
├── curated_ponzi_test.csv              60-message multilingual test set
├── telegram_data.csv                   Combined training corpus (~15k messages)
├── stage2_finetuned_results.csv        Stage 2 evaluation output
├── stage3_alerts.csv                   Stage 3 structured alerts
├── ponzi_model/                        Fine-tuned XLM-RoBERTa-base weights
├── requirements.txt                    Python dependencies
└── README.md
```

---

## Dataset Sources

The combined training corpus is built from three thematically aligned sources:

1. **`thehamkercat/telegram-spam-ham`** (HuggingFace) — Telegram spam classification corpus
2. **`zefang-liu/phishing-email-dataset`** (HuggingFace) — labelled phishing emails
3. **`ucirvine/sms_spam`** (HuggingFace) — UCI SMS Spam Collection

Plus a hand-crafted supplement:

4. **`synthetic_ponzi_messages.csv`** — 401 Ponzi-specific messages across English, Nigerian Pidgin, Hindi, and Indonesian, modelled on documented real-world schemes (CBEX, MMM, Racksterli, MBA Forex, Loom Money, crypto mining bots)

Total corpus: ~15,000 messages, balanced 47% ponzi_related / 53% legitimate.

---

## Key Results

### Primary Evaluation (held-out test set, n=967)

| Configuration | Precision | Recall | F1 Score |
|---|---|---|---|
| Stage 1 (keyword filter) | 53.77% | 98.14% | 69.47% |
| Stage 2 zero-shot XLM-R | 54.42% | 17.66% | 26.67% |
| **Stage 2 fine-tuned XLM-RoBERTa-base** | **96.48%** | **97.82%** | **97.15%** |

### Curated Multilingual Evaluation (n=60)

| Configuration | Precision | Recall | F1 Score |
|---|---|---|---|
| Stage 2 alone | 55.56% | 100.00% | 71.43% |
| **Stage 2 + Stage 3 cascade** | **90.48%** | **63.33%** | **74.51%** |

Languages: English, Nigerian Pidgin, Hindi, Indonesian
Per-language recall: 100% across all four languages.

---

## Architecture Notes

This is a **pilot implementation** of the production framework specified
in Chapter 3. Pilot substitutions:

| Component | Chapter 3 (Production) | Methodology Test Implementation(This Repo) |
|---|---|---|
| Stage 1 | L1 logistic regression, 220 features | 100-term keyword filter |
| Stage 2 | 4-transformer ensemble (XLM-R-Large + AfroXLMR + IndicBERT + mT5+LoRA) | Single XLM-RoBERTa-base multilingual |
| Stage 3 | 15-marker neural reasoner, per-country calibration | 12 weighted rule categories |
| Corpus | 58,214 regulator-verified posts | ~15,000 multi-source messages |
| Infrastructure | 64 A100 GPUs | Single CPU workstation |

The production specification is carried forward as recommendations in
Chapter 5 of the thesis.

---

## License & Citation

Academic project — see the thesis document for full methodology and references.

If you use this work, please cite:

```
Somide, O. (2026). PonziGuard: A Generalizable, Real-Time Detector of
Social-Media Ponzi Schemes. Final Year Project, Department of Computer, Pan-Atlantic University.
```

---

## Acknowledgments
- HuggingFace for hosting the base datasets
- The Streamlit team for the dashboard framework
