"""
PonziGuard - Stage 2: Fine-tuning XLM-RoBERTa
Trains a binary classifier (ponzi_related / legitimate) on the
filtered Telegram pilot corpus produced by prepare_data.py
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────
MODEL_NAME    = "xlm-roberta-base"   # base (not large) — faster on CPU
MAX_LENGTH    = 128                  # token limit per message
BATCH_SIZE    = 16                   # reduce to 8 if you get memory errors
EPOCHS        = 3                    # 3 is enough for this dataset size
LEARNING_RATE = 2e-5
TEST_SIZE     = 0.2
VAL_SIZE      = 0.1
SEED          = 42
OUTPUT_DIR    = "./ponzi_model"

LABEL2ID = {"legitimate": 0, "ponzi_related": 1}
ID2LABEL = {0: "legitimate", 1: "ponzi_related"}

print("=" * 55)
print("  PonziGuard — Stage 2: Fine-tuning XLM-RoBERTa")
print("=" * 55)

# ── Load data ─────────────────────────────────────────────────────────
print("\nLoading dataset...")
df = pd.read_csv("telegram_data.csv")
df["text"]  = df["text"].fillna("")
df["label_id"] = df["label"].map(LABEL2ID)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)

print(f"Total messages:    {len(df):,}")
print(f"ponzi_related:     {(df['label'] == 'ponzi_related').sum():,}")
print(f"legitimate:        {(df['label'] == 'legitimate').sum():,}")

# ── Train / val / test split ──────────────────────────────────────────
train_val_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label_id"]
)
train_df, val_df = train_test_split(
    train_val_df, test_size=VAL_SIZE / (1 - TEST_SIZE),
    random_state=SEED, stratify=train_val_df["label_id"]
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df):,}")
print(f"  Val:   {len(val_df):,}")
print(f"  Test:  {len(test_df):,}")

# ── Tokenizer ─────────────────────────────────────────────────────────
print(f"\nLoading tokenizer for {MODEL_NAME}...")
print("(First run downloads ~1.1 GB — may take a few minutes)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded.")

# ── Dataset class ─────────────────────────────────────────────────────
class PonziDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

print("\nTokenizing datasets...")
train_dataset = PonziDataset(train_df["text"], train_df["label_id"], tokenizer, MAX_LENGTH)
val_dataset   = PonziDataset(val_df["text"],   val_df["label_id"],   tokenizer, MAX_LENGTH)
test_dataset  = PonziDataset(test_df["text"],  test_df["label_id"],  tokenizer, MAX_LENGTH)
print("Tokenization complete.")

# ── Model ─────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME} for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True
)
print("Model loaded.")

# ── Metrics function ──────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall":    recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1":        f1_score(labels, preds, pos_label=1, zero_division=0),
        "accuracy":  accuracy_score(labels, preds),
    }

# ── Training arguments ────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=50,
    seed=SEED,
    report_to="none",          # disable wandb/tensorboard
    use_cpu=not torch.cuda.is_available(),
)

# ── History callback: captures per-epoch training metrics ────────────
class HistoryCallback(TrainerCallback):
    """Captures training loss and validation metrics at every log step."""
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step, "epoch": state.epoch}
        entry.update(logs)
        self.history.append(entry)

history_cb = HistoryCallback()

# ── Trainer ───────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
        history_cb,
    ],
)

# ── Train ─────────────────────────────────────────────────────────────
print(f"\nStarting fine-tuning for {EPOCHS} epochs...")
print(f"Batch size: {BATCH_SIZE} | Learning rate: {LEARNING_RATE}")
print("This will take approximately 1-2 hours on CPU.\n")

trainer.train()

# ── Evaluate on test set ──────────────────────────────────────────────
print("\nEvaluating on held-out test set...")
test_output = trainer.predict(test_dataset)
test_preds  = np.argmax(test_output.predictions, axis=1)
test_labels = test_df["label_id"].values

# Probability scores for PR / ROC curves
test_probs = torch.softmax(
    torch.tensor(test_output.predictions), dim=-1
).numpy()[:, 1]

precision = precision_score(test_labels, test_preds, pos_label=1, zero_division=0)
recall    = recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
f1        = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)
accuracy  = accuracy_score(test_labels, test_preds)
cm        = confusion_matrix(test_labels, test_preds)

tp = cm[1][1]; fp = cm[0][1]
fn = cm[1][0]; tn = cm[0][0]

print("\n" + "=" * 55)
print("  STAGE 2 FINE-TUNED MODEL — TEST SET RESULTS")
print("=" * 55)
print(f"  True Positives:   {tp:,}")
print(f"  False Positives:  {fp:,}")
print(f"  False Negatives:  {fn:,}")
print(f"  True Negatives:   {tn:,}")
print(f"  Precision:        {precision:.2%}")
print(f"  Recall:           {recall:.2%}")
print(f"  F1 Score:         {f1:.2%}")
print(f"  Accuracy:         {accuracy:.2%}")
print("=" * 55)

# ── Save results ──────────────────────────────────────────────────────
test_df = test_df.copy()
test_df["stage2_pred_label"] = [ID2LABEL[p] for p in test_preds]
test_df["stage2_prob_ponzi"] = test_probs
test_df["stage2_correct"]    = test_preds == test_labels
test_df.to_csv("stage2_finetuned_results.csv", index=False)

# Save training history (for loss curve, metrics curve in dashboard)
history_df = pd.DataFrame(history_cb.history)
history_df.to_csv("training_history.csv", index=False)
print(f"  Training history saved to: training_history.csv ({len(history_df)} log entries)")

# Save model and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n  Model saved to:   {OUTPUT_DIR}/")
print(f"  Results saved to: stage2_finetuned_results.csv (with stage2_prob_ponzi column)")
print("\n  Fine-tuning complete. Run app.py to view updated dashboard.")