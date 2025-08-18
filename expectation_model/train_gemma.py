#!/usr/bin/env python3
import os
import math
import time
import torch
import torch.nn as nn
<<<<<<< HEAD
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.cuda import amp
=======
import torch.optim as optim
from torch import amp
>>>>>>> 7f95ac7 (updates)
from torch.utils.data import DataLoader
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import wandb
<<<<<<< HEAD
=======
from transformers import GPT2Model, GPT2TokenizerFast
>>>>>>> 7f95ac7 (updates)
from safetensors.torch import save_file

# -----------------------------
# Hyperparams & utils
# -----------------------------
def get_hyperparams():
    batch_size  = int(os.getenv("BATCH_SIZE", 512))
    num_workers = int(os.getenv("NUM_WORKERS", 4))
<<<<<<< HEAD
    epochs      = int(os.getenv("EPOCHS", 50))
    lr          = float(os.getenv("LEARNING_RATE", 5e-6))
    head_dim    = int(os.getenv("HEAD_HIDDEN_DIM", 256))
    val_split   = float(os.getenv("VAL_SPLIT", 0.1))
    max_len     = int(os.getenv("MAX_LEN", 256))
    grad_clip   = float(os.getenv("GRAD_CLIP", 1.0))
    l1_lambda   = float(os.getenv("L1_LAMBDA", 0.00001)) # New L1 sparsity regularization term
    return batch_size, num_workers, epochs, lr, head_dim, val_split, max_len, grad_clip, l1_lambda
=======
    epochs      = int(os.getenv("EPOCHS", 10))
    lr          = float(os.getenv("LEARNING_RATE", 1e-4))
    head_dim    = int(os.getenv("HEAD_HIDDEN_DIM", 128))
    val_split   = float(os.getenv("VAL_SPLIT", 0.1))
    max_len     = int(os.getenv("MAX_LEN", 256))
    grad_clip   = float(os.getenv("GRAD_CLIP", 1.0))
    return batch_size, num_workers, epochs, lr, head_dim, val_split, max_len, grad_clip
>>>>>>> 7f95ac7 (updates)

def first_present(cols, options):
    for c in options:
        if c in cols:
            return c
    raise ValueError(f"None of the text columns {options} found in dataset columns: {cols}")

# -----------------------------
# Model
# -----------------------------
class Query2SAE(nn.Module):
<<<<<<< HEAD
    def __init__(self, head_hidden_dim: int, sae_dim: int, layer_index: int = 12,
                 model_name: str = "google/gemma-2b-it"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True, torch_dtype = torch.bfloat16)
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze backbone

        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, sae_dim),
            nn.Softplus(beta=1.0)  # smooth nonnegativity
        )
        self.layer_index = layer_index

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            H = out.hidden_states[self.layer_index]                # [B, T, H] from Gemma block 12
            mask = attention_mask.unsqueeze(-1).float()            # [B, T, 1]
            pooled = (H * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # mean-pool -> [B, H]
        return self.head(pooled)                                   # [B, sae_dim] ≥ 0

=======
    def __init__(self, head_hidden_dim: int, sae_dim: int):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained("gpt2")
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze GPT-2
        self.head = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, sae_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        # no grad through backbone (we only train the head)
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = out.last_hidden_state[:, -1, :]  # [B, 768]
        return self.head(last_hidden)
>>>>>>> 7f95ac7 (updates)

# -----------------------------
# Metrics
# -----------------------------
def topk_overlap(preds, targets, k=50):
    predk = preds.topk(k, dim=1).indices           # [B, k]
    truek = targets.topk(k, dim=1).indices         # [B, k]
    hits = (predk.unsqueeze(2) == truek.unsqueeze(1)).sum(dim=(1, 2)).to(preds.dtype)  # [B]
    precision_at_k = (hits / k).mean().item()
    recall_at_k = precision_at_k  # equal-size sets
    hit_rate = (hits > 0).to(preds.dtype).mean().item()
    return precision_at_k, recall_at_k, hit_rate

def cosine_sim(preds, targets):
    preds_n = torch.nn.functional.normalize(preds, dim=1)
    targs_n = torch.nn.functional.normalize(targets, dim=1)
    return (preds_n * targs_n).sum(dim=1).mean().item()

@torch.no_grad()
<<<<<<< HEAD
def evaluate(model, loader, device, l1_lambda=None, k_list=(10, 50, 100)):
    model.eval()
    total_loss, total_cos_loss, total_sl1_loss, total_l1_loss, n_batches = 0.0, 0.0, 0.0, 0.0, 0
=======
def evaluate(model, loader, loss_fn, device, k_list=(10, 50, 100)):
    """Epoch-level evaluation: returns averages over the entire loader."""
    model.eval()
    total_loss, n_batches = 0.0, 0
>>>>>>> 7f95ac7 (updates)
    cos_list = []
    precs = {k: 0.0 for k in k_list}
    recs  = {k: 0.0 for k in k_list}
    hits  = {k: 0.0 for k in k_list}

<<<<<<< HEAD
    smoothl1 = nn.SmoothL1Loss(beta=0.5)

=======
>>>>>>> 7f95ac7 (updates)
    for batch in tqdm(loader, desc="Eval", leave=False):
        x_ids  = batch["input_ids"].to(device, non_blocking=True)
        x_mask = batch["attention_mask"].to(device, non_blocking=True)
        y      = batch["sae_acts"].to(device, non_blocking=True).float()

        preds = model(x_ids, attention_mask=x_mask)
<<<<<<< HEAD

        cos_loss = 1.0 - F.cosine_similarity(preds + 1e-8, y + 1e-8, dim=1).mean()
        sl1_loss = smoothl1(preds, y)
        l1_loss  = preds.abs().mean()
        loss = 0.3 * cos_loss + 0.7 * sl1_loss + (l1_lambda or 0.0) * l1_loss

        total_loss     += loss.item()
        total_cos_loss += cos_loss.item()
        total_sl1_loss += sl1_loss.item()
        total_l1_loss  += l1_loss.item()
=======
        loss = loss_fn(preds, y)
        total_loss += loss.item()
>>>>>>> 7f95ac7 (updates)
        n_batches += 1

        cos_list.append(cosine_sim(preds, y))
        for k in k_list:
            p, r, h = topk_overlap(preds, y, k=k)
<<<<<<< HEAD
            precs[k] += p; recs[k] += r; hits[k] += h

    avg = lambda x: x / max(n_batches, 1)
    metrics = {
        "TotalLoss":   avg(total_loss),
        "CosineLoss":  avg(total_cos_loss),
        "SmoothL1Loss":avg(total_sl1_loss),
        "L1Loss":      avg(total_l1_loss),
        "Cosine":      sum(cos_list) / max(len(cos_list), 1),
=======
            precs[k] += p
            recs[k]  += r
            hits[k]  += h

    avg_loss = total_loss / max(n_batches, 1)
    avg_cos  = sum(cos_list) / max(len(cos_list), 1)
    metrics = {
        "Loss": avg_loss,
        "Cosine": avg_cos,
>>>>>>> 7f95ac7 (updates)
    }
    for k in k_list:
        metrics[f"Precision@{k}"] = precs[k] / max(n_batches, 1)
        metrics[f"Recall@{k}"]    = recs[k]  / max(n_batches, 1)
        metrics[f"HitRate@{k}"]   = hits[k]  / max(n_batches, 1)
    return metrics

<<<<<<< HEAD

=======
>>>>>>> 7f95ac7 (updates)
# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    (batch_size, num_workers, epochs, lr, head_dim,
<<<<<<< HEAD
     val_split, max_len, grad_clip, l1_lambda) = get_hyperparams()
=======
     val_split, max_len, grad_clip) = get_hyperparams()
>>>>>>> 7f95ac7 (updates)

    run = wandb.init(
        entity="manikx",
        project="hallucination_circuits",
        config={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "epochs": epochs,
            "learning_rate": lr,
            "head_hidden_dim": head_dim,
            "val_split": val_split,
            "max_len": max_len,
            "grad_clip": grad_clip,
<<<<<<< HEAD
            "l1_lambda": l1_lambda, # Added L1 lambda to wandb config
            "backbone": "google/gemma-2b-it (block 12)",
=======
            "backbone": "gpt2",
>>>>>>> 7f95ac7 (updates)
            "dataset": os.getenv("HF_DATASET_ID", "mksethi/eli5-gemma-features"),
        }
    )

    # --- W&B axes: use custom x-axes rather than the run step
    wandb.define_metric("global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("TrainStep/*", step_metric="global_step")  # per-batch charts
    wandb.define_metric("Train/*",     step_metric="epoch")        # per-epoch charts
    wandb.define_metric("Val/*",       step_metric="epoch")        # per-epoch eval charts
<<<<<<< HEAD
=======
    # (No use of wandb.log(step=...), which avoids monotonic step warnings.)  # ref docs
    # https://docs.wandb.ai/guides/track/log/customize-logging-axes/  # noqa
>>>>>>> 7f95ac7 (updates)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # 1) Tokenizer
<<<<<<< HEAD

    model_name = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

=======
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
>>>>>>> 7f95ac7 (updates)
    pad_id = tokenizer.pad_token_id

    # 2) Load HF dataset (mksethi/eli5-gemma-features)
    hf_id = os.getenv("HF_DATASET_ID", "mksethi/eli5-gemma-features")
    ds_all = load_dataset(hf_id)

    # pick splits or create a val split
    if "train" in ds_all:
        train_ds = ds_all["train"]
        val_ds = ds_all.get("validation") or ds_all.get("val")
        if val_ds is None:
            split = train_ds.train_test_split(test_size=val_split, seed=42)
            train_ds, val_ds = split["train"], split["test"]
    else:
        only = next(iter(ds_all.values()))
        split = only.train_test_split(test_size=val_split, seed=42)
        train_ds, val_ds = split["train"], split["test"]

    # 3) Tokenize from text column
    text_col = first_present(train_ds.column_names,
                             ["answer_text", "input", "query", "question", "text", "prompt"])

    def tok_fn(batch):
        toks = tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }

    keep_cols = set([text_col, "sae_acts"])
    remove_cols_train = [c for c in train_ds.column_names if c not in keep_cols]
    remove_cols_val   = [c for c in val_ds.column_names   if c not in keep_cols]

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=remove_cols_train, desc="Tokenizing train")
    val_ds   = val_ds.map(tok_fn,   batched=True, remove_columns=remove_cols_val,   desc="Tokenizing val")

    # 4) set torch format
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "sae_acts"])
    val_ds.set_format(type="torch",   columns=["input_ids", "attention_mask", "sae_acts"])

    # 5) DataLoaders
    optimal_workers = min(num_workers, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=optimal_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # 6) Build model
    sae_dim = len(train_ds[0]["sae_acts"])
<<<<<<< HEAD
    model = Query2SAE(head_hidden_dim=head_dim, sae_dim=sae_dim,
                  layer_index=12, model_name=model_name).to(device)


    # 7) Optimizer / loss / AMP
    optimizer = optim.AdamW(model.head.parameters(), lr=lr)
    smoothl1 = nn.SmoothL1Loss(beta=0.5)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

=======
    model = Query2SAE(head_hidden_dim=head_dim, sae_dim=sae_dim).to(device)

    # 7) Optimizer / loss / AMP
    optimizer = optim.AdamW(model.head.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    scaler = amp.GradScaler("cuda", enabled=torch.cuda.is_available())  # new API
>>>>>>> 7f95ac7 (updates)

    wandb.watch(model.head, log="all", log_freq=50)

    # 8) Training
    ckpt_dir = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = math.inf

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
<<<<<<< HEAD
        total_loss_running = 0.0
=======
        total_loss = 0.0
        prec10 = prec50 = prec100 = 0.0
        rec10 = rec50 = rec100 = 0.0
        hits50 = 0.0
>>>>>>> 7f95ac7 (updates)
        n_batches = 0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x_ids  = batch["input_ids"].to(device, non_blocking=True)
            x_mask = batch["attention_mask"].to(device, non_blocking=True)
            y      = batch["sae_acts"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
<<<<<<< HEAD
            with amp.autocast(enabled=torch.cuda.is_available()):
                preds = model(x_ids, attention_mask=x_mask)
                cos_loss = 1.0 - F.cosine_similarity(preds + 1e-8, y + 1e-8, dim=1).mean()
                sl1_loss = smoothl1(preds, y)
                l1_loss  = preds.abs().mean()
                loss = 0.3 * cos_loss + 0.7 * sl1_loss + l1_lambda * l1_loss
=======
            with amp.autocast("cuda", enabled=torch.cuda.is_available()):  # new API
                preds = model(x_ids, attention_mask=x_mask)
                loss  = loss_fn(preds, y)
>>>>>>> 7f95ac7 (updates)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # running tallies
<<<<<<< HEAD
            total_loss_running += loss.item()
            n_batches += 1

            # per-batch logs
            wandb.log({
                "global_step": global_step,
                "TrainStep/LR": optimizer.param_groups[0]["lr"],
                "TrainStep/TotalLoss": float(loss.item()),
                "TrainStep/CosineLoss": float(cos_loss.item()),
                "TrainStep/SmoothL1Loss": float(sl1_loss.item()),
                "TrainStep/L1Loss": float(l1_loss.item()),
            })
            global_step += 1

        # epoch-level train metrics
        train_loss = total_loss_running / max(n_batches, 1)
        wandb.log({
            "epoch": epoch,
            "Train/TotalLoss": train_loss,
            "Train/CosineLoss": float(cos_loss.item()),
            "Train/SmoothL1Loss": float(sl1_loss.item()),
            "Train/L1Loss": float(l1_loss.item()),
=======
            total_loss += loss.item()
            n_batches += 1

            # per-batch logs (NO cosine here by request)
            wandb.log({
                "global_step": global_step,
                "TrainStep/Loss": float(loss.item()),
                "TrainStep/LR": optimizer.param_groups[0]["lr"],
            })
            global_step += 1

            # accumulate train @k for epoch-level view (optional)
            with torch.no_grad():
                p10, r10, _   = topk_overlap(preds, y, k=10)
                p50, r50, h50 = topk_overlap(preds, y, k=50)
                p100, r100, _ = topk_overlap(preds, y, k=100)
                prec10 += p10; rec10 += r10
                prec50 += p50; rec50 += r50; hits50 += h50
                prec100 += p100; rec100 += r100

        # epoch-level train metrics
        train_loss = total_loss / max(n_batches, 1)
        wandb.log({
            "epoch": epoch,
            "Train/Loss": train_loss,
            "Train/Precision@10":  prec10 / max(n_batches, 1),
            "Train/Precision@50":  prec50 / max(n_batches, 1),
            "Train/Precision@100": prec100 / max(n_batches, 1),
            "Train/Recall@10":  rec10  / max(n_batches, 1),
            "Train/Recall@50":  rec50  / max(n_batches, 1),
            "Train/Recall@100": rec100 / max(n_batches, 1),
            "Train/HitRate@50": hits50 / max(n_batches, 1),
>>>>>>> 7f95ac7 (updates)
            "LR": optimizer.param_groups[0]["lr"],
            "EpochSecs": time.time() - t0,
        })

<<<<<<< HEAD
        
        # validation each epoch (with the combined loss)
        val_metrics = evaluate(model, val_loader, device=device, l1_lambda=l1_lambda) # The `evaluate` function now computes the composite loss internally
        wandb.log({"epoch": epoch, **{f"Val/{k}": v for k, v in val_metrics.items()}})

        # optional: generalization gap
        if "TotalLoss" in val_metrics:
            wandb.log({"Val/GenGap": val_metrics["TotalLoss"] - train_loss, "epoch": epoch})
=======
        # validation each epoch (includes epoch-avg Cosine)
        val_metrics = evaluate(model, val_loader, loss_fn, device, k_list=(10, 50, 100))
        wandb.log({"epoch": epoch, **{f"Val/{k}": v for k, v in val_metrics.items()}})


        # optional: generalization gap
        if "Loss" in val_metrics:
            wandb.log({"Val/GenGap": val_metrics["Loss"] - train_loss, "epoch": epoch})
>>>>>>> 7f95ac7 (updates)

        # Save per-epoch checkpoint
        st_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.safetensors")
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(cpu_state, st_path)
        print(f"→ Saved checkpoint: {st_path}")

<<<<<<< HEAD
        # Track best by Val/TotalLoss
        if val_metrics["TotalLoss"] < best_val:
            best_val = val_metrics["TotalLoss"]
            best_path = os.path.join(ckpt_dir, "model_best.safetensors")
            save_file(cpu_state, best_path)
            print(f"✓ New best (Val/TotalLoss={best_val:.4f}). Saved: {best_path}")

    # 9) Final eval and save "last"
    final_metrics = evaluate(model, val_loader, device=device, l1_lambda=l1_lambda)
=======
        # Track best by Val/Loss
        if val_metrics["Loss"] < best_val:
            best_val = val_metrics["Loss"]
            best_path = os.path.join(ckpt_dir, "model_best.safetensors")
            save_file(cpu_state, best_path)
            print(f"✓ New best (Val/Loss={best_val:.4f}). Saved: {best_path}")

    # 9) Final eval and save "last"
    final_metrics = evaluate(model, val_loader, loss_fn, device, k_list=(10, 50, 100))
>>>>>>> 7f95ac7 (updates)
    print("Final Validation:", final_metrics)
    wandb.log({f"Val_Final/{k}": v for k, v in final_metrics.items()})

    # Summaries
<<<<<<< HEAD
    wandb.run.summary["Best/ValTotalLoss"] = best_val
=======
    wandb.run.summary["Best/ValLoss"] = best_val
>>>>>>> 7f95ac7 (updates)
    for k, v in final_metrics.items():
        wandb.run.summary[f"Final/{k}"] = v

    last_path = os.path.join(ckpt_dir, "model_last.safetensors")
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(cpu_state, last_path)
    print(f"→ Saved last checkpoint: {last_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
