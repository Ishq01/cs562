# %%
import torch, gc

gc.collect()
torch.cuda.empty_cache()


# %%
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE, "CUDA index:", torch.cuda.current_device())
print("GPU name:", torch.cuda.get_device_name())


# %%
# cell 1
import os
import json
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    DebertaV2ForQuestionAnswering,
)
from datasets import load_dataset
from transformers import default_data_collator

from opacus.grad_sample import GradSampleModule
from opacus.accountants import RDPAccountant

from tqdm import tqdm

# -------------------
# Device & reproducibility
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------
# Paths
# -------------------
CKPT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# -------------------
# Training hyperparameters
# -------------------
MODEL_NAME = "microsoft/deberta-v3-base"

EPOCHS = 5
LR = 2e-5

TRAIN_BATCH_SIZE = 8   # you can tune this
EVAL_BATCH_SIZE = 8

# -------------------
# DP hyperparameters (project ones)
# -------------------
NOISE_MULTIPLIER = 1.0   # σ
MAX_GRAD_NORM   = 1.0    # C
DELTA           = 1e-5   # δ

print("Config set.")

# %%
# cell 1.5 
# Progress

import torch
from pathlib import Path

CHECKPOINT_PATH = CKPT_DIR / "dp_qa_checkpoint.pt"

def save_dp_checkpoint(path, dp_model, optimizer, accountant, history, epoch):
    """
    Save DP training state so we can resume later.
    """
    ckpt = {
        "epoch": epoch,
        "model_state": dp_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "accountant_state": accountant.state_dict(),  # RDPAccountant supports state_dict
        "history": history,
    }
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved at epoch {epoch} → {path}")


def load_dp_checkpoint(path, dp_model, optimizer, accountant, device=DEVICE):
    """
    Load DP training state. Returns (start_epoch, history).
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,  # IMPORTANT for PyTorch 2.6+ when loading arbitrary Python objects
    )

    dp_model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    accountant.load_state_dict(ckpt["accountant_state"])
    history = ckpt.get("history", [])
    start_epoch = ckpt["epoch"] + 1

    print(f"[Checkpoint] Loaded from epoch {ckpt['epoch']} → resume at epoch {start_epoch}")
    return start_epoch, history

# %%
# cell 2
# Load the HotPotQA "distractor" split
dataset = load_dataset("hotpot_qa", "distractor")

train_raw = dataset["train"]
val_raw   = dataset["validation"]

print("Original sizes → Train:", len(train_raw), "Val:", len(val_raw))

# -----------------------------
# Reduce to 1/10th for testing
# -----------------------------
# fraction = 0.1 means: keep only 10%
frac = 1

train_raw = train_raw.select(range(int(len(train_raw) * frac)))
val_raw   = val_raw.select(range(int(len(val_raw) * frac)))

print("Reduced sizes → Train:", len(train_raw), "Val:", len(val_raw))

# %%
# Cell 3 – Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded:", MODEL_NAME)

# %%
# Cell 4 – Preprocess function (flatten context + align spans)

# This is essentially your earlier preprocess:

def preprocess(example):
    """
    Convert HotPotQA example into extractive QA features.

    HotPotQA "distractor" has:
      - example["question"]
      - example["answer"] (string)
      - example["context"]["sentences"]: list of list-of-sentences

    We:
      - flatten all sentences into one long context string
      - find answer char span
      - map char span -> token span using offset_mapping
    """

    # Flatten nested list of sentences
    nested_sents = example["context"]["sentences"]  # [num_docs][num_sents_per_doc]
    flat_sentences = [sent for doc in nested_sents for sent in doc]

    context = " ".join(flat_sentences)
    question = example["question"]
    answer_text = example["answer"]   # string

    if not isinstance(answer_text, str) or len(answer_text.strip()) == 0:
        return None

    # Find character span of the answer in the context (case-insensitive)
    ans_start = context.lower().find(answer_text.lower())
    if ans_start == -1:
        return None  # answer not found in context

    ans_end = ans_start + len(answer_text)

    # Tokenize with offsets for span mapping
    encoded = tokenizer(
        question,
        context,
        truncation=True,
        max_length=512, # ORIGINAL 512
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]
    start_tok = None
    end_tok = None

    # Convert character → token spans
    for i, (s, e) in enumerate(offsets):
        if s <= ans_start < e:
            start_tok = i
        if s < ans_end <= e:
            end_tok = i

    if start_tok is None or end_tok is None:
        return None

    encoded["start_positions"] = start_tok
    encoded["end_positions"] = end_tok

    encoded["answer_text"] = answer_text

    # Remove offsets — not needed for model forward()
    encoded.pop("offset_mapping")

    return encoded

# %%
# Cell 5 – Run preprocessing to create processed lists

# Run preprocessing over train & val
train_processed = []
train_answers = []

for ex in tqdm(train_raw, desc="Train preprocess"):
    out = preprocess(ex)
    if out:
        train_answers.append(out["answer_text"])
        # remove answer_text from features stored in dataset
        feat = {k: v for k, v in out.items() if k != "answer_text"}
        train_processed.append(feat)

val_processed = []
val_answers = []

for ex in tqdm(val_raw, desc="Val preprocess"):
    out = preprocess(ex)
    if out:
        val_answers.append(out["answer_text"])
        feat = {k: v for k, v in out.items() if k != "answer_text"}
        val_processed.append(feat)

print("Train processed:", len(train_processed))
print("Val processed:", len(val_processed))


# %%
# Cell 6 – Dataset wrapper
class QADataset(Dataset):
    """
    Simple dataset wrapper for processed QA features.
    Each item contains:
      - input_ids
      - attention_mask
      - start_positions
      - end_positions
    """

    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # IMPORTANT: keep ids/masks as plain lists; collator will pad & tensorize
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "start_positions": torch.tensor(item["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(item["end_positions"], dtype=torch.long),
        }

# Build dataset objects
train_dataset = QADataset(train_processed)
val_dataset   = QADataset(val_processed)

print("Dataset sizes → Train:", len(train_dataset), "Val:", len(val_dataset))

# %%
# Cell 7 – DataLoaders
from transformers import DataCollatorWithPadding

# Dynamic padding collator: pads to longest sequence in the batch
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,          # pad to max length in each batch
    return_tensors="pt",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    drop_last=True,   # helpful for DP accounting
)

val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    drop_last=False,
)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

# %%
# Cell 8 – Build DP-ready DeBERTa model

# We disable gradient checkpointing and wrap with GradSampleModule
def make_dp_model():
    """
    Create a fresh DeBERTa QA model wrapped with GradSampleModule for DP.
    """
    base_model = DebertaV2ForQuestionAnswering.from_pretrained(MODEL_NAME)
    base_model.to(DEVICE)
    base_model.train()

    # IMPORTANT: disable gradient checkpointing when using Opacus GradSampleModule
    base_model.gradient_checkpointing_disable()

    dp_model = GradSampleModule(base_model).to(DEVICE)
    dp_model.train()
    return dp_model

# dp_model_test = make_dp_model()
print("DP-ready model created.")

# %%
# Cell 9 – Span metrics (EM & F1 on token indices)
import re
import string

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace (SQuAD-style)."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def string_f1(prediction: str, ground_truth: str) -> float:
    """F1 over word tokens of normalized strings."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0

    # Count overlaps properly
    from collections import Counter
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    overlap = sum(min(pred_counts[w], gt_counts[w]) for w in common)

    precision = overlap / len(pred_tokens)
    recall    = overlap / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def string_em(prediction: str, ground_truth: str) -> float:
    """Exact match on normalized strings."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# %%
# Cell 9.5 String based evaluation
def evaluate_string_metrics(model, data_loader, gold_answers):
    """
    Evaluate model using string-based EM/F1 (HotPotQA/SQuAD-style):

      - Decode predicted span from token ids
      - Compare against gold answer string with normalization
    """
    model.eval()
    total = 0
    em_sum = 0.0
    f1_sum = 0.0

    idx = 0  # global index over examples in val_answers

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            start_logits = outputs.start_logits
            end_logits   = outputs.end_logits

            start_pred = start_logits.argmax(dim=-1)
            end_pred   = end_logits.argmax(dim=-1)

            input_ids = batch["input_ids"]  # [B, T]
            batch_size = input_ids.size(0)

            for i in range(batch_size):
                # Clamp end_pred >= start_pred
                s = int(start_pred[i].item())
                e = int(end_pred[i].item())
                if e < s:
                    e = s

                # Decode span to text
                span_ids = input_ids[i, s:e+1]
                pred_text = tokenizer.decode(span_ids, skip_special_tokens=True).strip()

                gold_text = gold_answers[idx]

                em = string_em(pred_text, gold_text)
                f1 = string_f1(pred_text, gold_text)

                em_sum += em
                f1_sum += f1
                total += 1
                idx += 1

    em = em_sum / max(total, 1)
    f1 = f1_sum / max(total, 1)
    return em, f1


# %%
# Cell 10 – One DP-SGD step (generalization of your Cell E)

# This is exactly your per-batch DP clip+noise, wrapped in a function.
def dp_sgd_step(dp_model, optimizer, batch, C, noise_multiplier):
    """
    Perform one DP-SGD update step:
      - forward pass
      - backward to populate .grad_sample
      - per-sample clipping + Gaussian noise
      - optimizer.step()

    Returns: scalar loss (float)
    """
    dp_model.train()
    optimizer.zero_grad()

    # Move batch to device
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    # Forward
    outputs = dp_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        start_positions=batch["start_positions"],
        end_positions=batch["end_positions"],
    )
    loss = outputs.loss

    # Backward: populate .grad_sample
    loss.backward()

    # DP clipping + noise
    num = 0
    for p in dp_model.parameters():
        if not hasattr(p, "grad_sample") or p.grad_sample is None:
            continue

        g_sample = p.grad_sample           # [batch_size, ...]
        B = g_sample.shape[0]

        # Flatten per-sample grads to compute norms
        g_flat = g_sample.view(B, -1)
        norms = g_flat.norm(2, dim=1)      # [B]

        # Compute clipping factors
        clip_factors = (C / (norms + 1e-6)).clamp(max=1.0)  # [B]

        # Broadcast to parameter shape
        view_shape = [B] + [1] * (g_sample.dim() - 1)
        g_clipped = g_sample * clip_factors.view(*view_shape)

        # Aggregate clipped grads
        g_sum = g_clipped.sum(dim=0)

        # Add Gaussian noise
        noise = torch.normal(
            mean=0.0,
            std=noise_multiplier * C,
            size=p.shape,
            device=p.device,
            dtype=p.dtype,
        )

        # Final DP gradient
        p.grad = g_sum + noise

        # Free per-sample grads
        p.grad_sample = None
        num += 1

    optimizer.step()

    return float(loss.item())

# %%
# Cell 11 – Full DP training loop with RDP accountant

# This trains over the whole dataset and tracks ε per epoch.
def train_dp_model(
    train_loader,
    val_loader,
    epochs=EPOCHS,
    noise_multiplier=NOISE_MULTIPLIER,
    max_grad_norm=MAX_GRAD_NORM,
    lr=LR,
    delta=DELTA,
    batch_size=TRAIN_BATCH_SIZE,   # must match loader creation
    start_epoch=1,                 # NEW: for resume
    dp_model=None,                 # NEW: pass in model when resuming
    optimizer=None,                # NEW: pass in optimizer when resuming
    accountant=None,               # NEW: pass in accountant when resuming
    history=None,                  # NEW: existing history when resuming
    checkpoint_path=CHECKPOINT_PATH,  # where to save checkpoints
):
    """
    Train DeBERTa QA model with manual DP-SGD over the whole train set.

    Returns: (dp_model, history)
      - dp_model: trained model
      - history: list of dicts with metrics per epoch
    """

    # --- Initialize / reuse model, optimizer, accountant, history ---
    if dp_model is None:
        dp_model = make_dp_model()
    if optimizer is None:
        optimizer = AdamW(dp_model.parameters(), lr=lr)
    if accountant is None:
        accountant = RDPAccountant()
    if history is None:
        history = []

    # Sample rate for DP accounting (Poisson subsampling approximation)
    dataset_size = len(train_loader.dataset)
    sample_rate = batch_size / dataset_size

    print(
        f"Dataset size: {dataset_size}, batch size: {batch_size}, "
        f"sample_rate: {sample_rate:.6f}"
    )

    # Main epoch loop
    for epoch in range(start_epoch, epochs + 1):
        dp_model.train()
        running_loss = 0.0
        num_batches = 0

        print(f"\n=== DP Epoch {epoch}/{epochs} ===")

        # tqdm progress bar for this epoch
        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch in epoch_iter:
            loss = dp_sgd_step(
                dp_model,
                optimizer,
                batch,
                C=max_grad_norm,
                noise_multiplier=noise_multiplier,
            )
            running_loss += loss
            num_batches += 1

            # One DP step in the accountant
            accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
            )

            # Update progress bar
            if num_batches > 0:
                avg_loss_so_far = running_loss / num_batches
                epoch_iter.set_postfix(loss=f"{avg_loss_so_far:.4f}")

        avg_loss = running_loss / max(num_batches, 1)
        epsilon = accountant.get_epsilon(delta)  # single float in your Opacus

        # Evaluate on validation set with string-based EM/F1
        em, f1 = evaluate_string_metrics(dp_model, val_loader, val_answers)

        print(
            f"Epoch {epoch}: loss={avg_loss:.4f}, "
            f"Îµ={epsilon:.3f}, EM={em:.4f}, F1={f1:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "avg_loss": avg_loss,
                "epsilon": float(epsilon),
                "em": float(em),
                "f1": float(f1),
                "noise": float(noise_multiplier),
                "C": float(max_grad_norm),
                "lr": float(lr),
                "batch_size": int(batch_size),
            }
        )

        # --- Save checkpoint at the end of each epoch ---
        if checkpoint_path is not None:
            save_dp_checkpoint(
                path=checkpoint_path,
                dp_model=dp_model,
                optimizer=optimizer,
                accountant=accountant,
                history=history,
                epoch=epoch,
            )


    return dp_model, history

# %%
# # Cell 12 – Run or resume one DP experiment

# from pathlib import Path

# # Fresh placeholders
# dp_model = None
# optimizer = None
# accountant = None
# history = None
# start_epoch = 1

# if CHECKPOINT_PATH.exists():
#     print(f"Checkpoint found at {CHECKPOINT_PATH}. Loading and resuming...")
#     # Create empty model/optimizer/accountant, then load into them
#     dp_model = make_dp_model()
#     optimizer = AdamW(dp_model.parameters(), lr=LR)
#     accountant = RDPAccountant()

#     start_epoch, history = load_dp_checkpoint(
#         CHECKPOINT_PATH,
#         dp_model,
#         optimizer,
#         accountant,
#         device=DEVICE,
#     )
# else:
#     print("No checkpoint found. Starting fresh training.")
#     # dp_model, optimizer, accountant, history remain None
#     start_epoch = 1

# # Train (fresh or resumed)
# dp_model, dp_history = train_dp_model(
#     train_loader=train_loader,
#     val_loader=val_loader,
#     epochs=EPOCHS,             # total epochs you want
#     noise_multiplier=NOISE_MULTIPLIER,
#     max_grad_norm=MAX_GRAD_NORM,
#     lr=LR,
#     delta=DELTA,
#     batch_size=TRAIN_BATCH_SIZE,
#     start_epoch=start_epoch,   # will be 1 for fresh, >1 for resume
#     dp_model=dp_model,
#     optimizer=optimizer,
#     accountant=accountant,
#     history=history,
#     checkpoint_path=CHECKPOINT_PATH,
# )

# print("\nDP training complete. History entries:")
# for h in dp_history:
#     print(h)

# %%
# Cell 13 - result directory + helpers

import os
import json
import csv
from pathlib import Path
import itertools
import datetime

RESULTS_DIR = Path("dp_results")
RESULTS_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = RESULTS_DIR / "dp_runs_summary.csv"

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def config_to_run_id(config):
    """Create a short identifier string for a run from its config."""
    return (
        f"sigma{config['sigma']}_"
        f"C{config['C']}_"
        f"bs{config['batch_size']}_"
        f"ep{config['epochs']}"
    )

def save_run_history(run_id, config, history):
    """
    Save full history (all epochs) for a single run as JSON.
    File name encodes hyperparameters for easier inspection.
    """
    out_path = RESULTS_DIR / f"history_{run_id}.json"
    payload = {
        "config": config,
        "history": history,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved history] {out_path}")

def save_config_checkpoint(run_id, epoch, dp_model, optimizer, accountant, history):
    ckpt_path = RESULTS_DIR / f"ckpt_{run_id}.pt"
    torch.save({
        "epoch": epoch,
        "model": dp_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "accountant": accountant.states if hasattr(accountant, "states") else None,
        "history": history,
    }, ckpt_path)
    print(f"[Checkpoint saved] {ckpt_path}")

def load_config_checkpoint(run_id, DEVICE):
    ckpt_path = RESULTS_DIR / f"ckpt_{run_id}.pt"
    if not ckpt_path.exists():
        return None

    data = torch.load(ckpt_path, map_location=DEVICE)

    return data


def append_summary_rows(run_id, config, history):
    """
    Append per-epoch rows to a CSV summary file:
    one row per epoch per run.
    """
    # Make sure the file has a header the first time
    file_exists = SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "run_id",
                    "timestamp",
                    "sigma",
                    "C",
                    "batch_size",
                    "epochs_total",
                    "epoch",
                    "avg_loss",
                    "epsilon",
                    "em",
                    "f1",
                    "lr",
                    "delta",
                    "dataset_size",
                ]
            )
        ts = get_timestamp()
        dataset_size = len(train_dataset)
        for h in history:
            writer.writerow(
                [
                    run_id,
                    ts,
                    config["sigma"],
                    config["C"],
                    config["batch_size"],
                    config["epochs"],
                    h["epoch"],
                    h["avg_loss"],
                    h["epsilon"],
                    h["em"],
                    h["f1"],
                    config["lr"],
                    config["delta"],
                    dataset_size,
                ]
            )
    print(f"[Updated summary CSV] {SUMMARY_CSV}")


# %%
# Cell 13.5 - resume from where it left off
def get_completed_runs(epochs_required=None):
    """
    Scan RESULTS_DIR for history_*.json and return a dict:
      run_id -> (config, history)

    If epochs_required is not None, we only consider a run 'completed'
    if its last epoch >= epochs_required.
    """
    completed = {}

    for path in RESULTS_DIR.glob("history_*.json"):
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"[Warning] Could not load {path}: {e}")
            continue

        config = payload.get("config", {})
        history = payload.get("history", [])
        if not history:
            continue

        last_epoch = history[-1].get("epoch", 0)
        run_id = path.stem.replace("history_", "")

        if epochs_required is not None and last_epoch < epochs_required:
            # This run did not finish all epochs; treat as incomplete
            continue

        completed[run_id] = (config, history)

    return completed

# %%
# Cell 14 - Runner for a single configuration

from torch.utils.data import DataLoader

def run_single_dp_config(
    sigma,
    C,
    batch_size,
    epochs,
    lr=LR,
    delta=DELTA,
    eval_batch_size=EVAL_BATCH_SIZE,
):
    """
    Run one DP training configuration and log results.

    Returns: (run_id, history)
    """
    config = {
        "sigma": float(sigma),
        "C": float(C),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "delta": float(delta),
    }

    run_id = config_to_run_id(config)
    print(f"\n=== Starting run: {run_id} ===")

    # Recreate loaders for this batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        drop_last=False,
    )

    existing = load_config_checkpoint(run_id, DEVICE)

    if existing is not None:
        print(f"[Resume] Found checkpoint for {run_id}, epoch {existing['epoch']}. Resuming ...")
        start_epoch = existing["epoch"] + 1
        history = existing["history"]

        # rebuild model, optimizer, accountant
        dp_model = build_new_dp_model()           # however you normally construct it
        dp_model.load_state_dict(existing["model"])

        optimizer = torch.optim.AdamW(dp_model.parameters(), lr=config["lr"])
        optimizer.load_state_dict(existing["optimizer"])

        accountant = RDPAccountant()
        if existing["accountant"] is not None:
            accountant.states = existing["accountant"]

    else:
        # starting from scratch
        start_epoch = 1
        history = []
        dp_model = build_new_dp_model()
        optimizer = torch.optim.AdamW(dp_model.parameters(), lr=config["lr"])
        accountant = RDPAccountant()

    dp_model, history = train_dp_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        noise_multiplier=config["sigma"],
        max_grad_norm=config["C"],
        lr=config["lr"],
        delta=config["delta"],
        batch_size=config["batch_size"],
        start_epoch=start_epoch,
        dp_model=dp_model,
        optimizer=optimizer,
        accountant=accountant,
        history=history,
        checkpoint_path=None,   # we use our per-run checkpoint instead
        run_id=run_id,          # pass run ID to save per-epoch checkpoints
    )

    # Save history & update summary
    save_run_history(run_id, config, history)
    append_summary_rows(run_id, config, history)

    # Print final epoch
    last = history[-1]
    print(
        f"[Run complete] {run_id} | "
        f"final epoch={last['epoch']}, "
        f"ε={last['epsilon']:.3f}, EM={last['em']:.4f}, F1={last['f1']:.4f}"
    )

    return run_id, history


# %%
def run_dp_sweep(
    sigma_list,
    C_list,
    batch_list,
    epochs,
    lr=LR,
    delta=DELTA,
):
    """
    Run a grid of DP configurations with full resume support.

    Behavior:
      - For each (sigma, C, batch_size) config, we derive a run_id and a checkpoint path.
      - If a checkpoint exists and its epoch >= `epochs`, we SKIP training and just load history.
      - If a checkpoint exists but its epoch < `epochs`, we RESUME from the next epoch.
      - If no checkpoint exists, we start training from epoch 1.

    Assumes:
      - train_dp_model(...) saves checkpoints to the given checkpoint_path
        at the end of each epoch via save_dp_checkpoint.
      - load_dp_checkpoint(...) can load that checkpoint and return
        (start_epoch, history).
    """
    all_runs = []

    for sigma, C, bs in itertools.product(sigma_list, C_list, batch_list):
        # Build config dict and run_id
        config = {
            "sigma": float(sigma),
            "C": float(C),
            "batch_size": int(bs),
            "epochs": int(epochs),
            "lr": float(lr),
            "delta": float(delta),
        }
        run_id = config_to_run_id(config)

        # Per-run checkpoint path (different file per config)
        ckpt_path = RESULTS_DIR / f"ckpt_{run_id}.pt"

        # ---- Decide whether to skip, resume, or start fresh ----
        dp_model = None
        optimizer = None
        accountant = None
        history = None
        start_epoch = 1

        if ckpt_path.exists():
            # There is some saved state for this config
            print(f"[Resume check] Found checkpoint for {run_id} at {ckpt_path}. Loading...")

            # Recreate empty model/optimizer/accountant and load
            dp_model = make_dp_model()
            optimizer = AdamW(dp_model.parameters(), lr=lr)
            accountant = RDPAccountant()

            start_epoch, history = load_dp_checkpoint(
                ckpt_path,
                dp_model,
                optimizer,
                accountant,
                device=DEVICE,
            )

            # If checkpoint already finished all requested epochs, just skip
            if start_epoch > epochs:
                print(f"[Skip] {run_id} already completed (last epoch={start_epoch-1} >= target={epochs}).")
                # history already loaded from checkpoint
                save_run_history(run_id, config, history)
                append_summary_rows(run_id, config, history)
                all_runs.append(
                    {
                        "run_id": run_id,
                        "sigma": sigma,
                        "C": C,
                        "batch_size": bs,
                        "epochs": epochs,
                        "history": history,
                    }
                )
                continue

            print(f"[Resume] {run_id}: resuming from epoch {start_epoch}/{epochs}.")
        else:
            print(f"[Run] {run_id}: no checkpoint found. Starting from scratch.")
            # start_epoch = 1, dp_model/optimizer/accountant/history = None
            # train_dp_model will initialize them

        # ---- Build DataLoaders for this config ----
        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            collate_fn=data_collator,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=data_collator,
            drop_last=False,
        )

        # ---- Run (or resume) training for this config ----
        dp_model, history = train_dp_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            noise_multiplier=sigma,
            max_grad_norm=C,
            lr=lr,
            delta=delta,
            batch_size=bs,
            start_epoch=start_epoch,       # <- resume from here
            dp_model=dp_model,             # None if fresh
            optimizer=optimizer,           # None if fresh
            accountant=accountant,         # None if fresh
            history=history,               # list if resumed, None if fresh
            checkpoint_path=ckpt_path,     # per-config checkpoint
        )

        # ---- Save final history & summary rows ----
        save_run_history(run_id, config, history)
        append_summary_rows(run_id, config, history)

        # Keep in memory for this Python run
        all_runs.append(
            {
                "run_id": run_id,
                "sigma": sigma,
                "C": C,
                "batch_size": bs,
                "epochs": epochs,
                "history": history,
            }
        )

        # Print final line for this config
        last = history[-1]
        print(
            f"[Run complete] {run_id} | "
            f"final epoch={last['epoch']}, "
            f"Îµ={last['epsilon']:.3f}, EM={last['em']:.4f}, F1={last['f1']:.4f}"
        )

    return all_runs


# %%
# Cell 16 - run configs

# Example: fix epochs=3 for the sweep to keep runtime manageable
sweep_results = run_dp_sweep(
    sigma_list=[.5, .75],       # noise multiplier
    C_list=[0.5, 1.0, 2.0],                     # clipping norm
    batch_list=[8],                             # batch size
    epochs=4,
)



