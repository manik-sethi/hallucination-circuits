#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, load_dataset_builder
from dotenv import load_dotenv
from tqdm import tqdm
import wandb
from transformers import GPT2Model, GPT2TokenizerFast
from safetensors.torch import save_file

def get_hyperparams():
    batch_size  = int(os.getenv("BATCH_SIZE", 512))
    num_workers = int(os.getenv("NUM_WORKERS", 4))
    epochs      = int(os.getenv("EPOCHS", 10))
    lr          = float(os.getenv("LEARNING_RATE", 1e-4))
    head_dim    = int(os.getenv("HEAD_HIDDEN_DIM", 128))
    val_split   = float(os.getenv("VAL_SPLIT", 0.1))
    return batch_size, num_workers, epochs, lr, head_dim, val_split

class Query2SAE(nn.Module):
    def __init__(self, head_hidden_dim: int, sae_dim: int):
        super().__init__()
        # 1) Load GPT‐2 backbone
        self.backbone = GPT2Model.from_pretrained("gpt2")
        # 2) Freeze it by default
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 3) A small head: [768 → head_hidden_dim → sae_dim]
        self.head = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, sae_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        # Pass through GPT2
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last token’s hidden state (you can also do mean pooling if you prefer)
        last_hidden = out.last_hidden_state[:, -1, :]  # (batch_size, 768)
        # Predict SAE vector
        return self.head(last_hidden)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, hit_count, n_seen = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            x_ids = batch["input_ids"].to(device)
            x_mask = batch["attention_mask"].to(device)
            y = batch["sae_features"].to(device).float()

            preds = model(x_ids, attention_mask=x_mask)
            loss = loss_fn(preds, y)
            total_loss += loss.item()

            pred50 = preds.topk(50, dim=1).indices
            true50 = y.topk(50, dim=1).indices
            hits = (pred50.unsqueeze(2) == true50.unsqueeze(1)).any(dim=(1,2))
            hit_count += hits.sum().item()
            n_seen    += hits.size(0)

    return total_loss / len(loader), hit_count / n_seen

def main():
    load_dotenv()
    batch_size, num_workers, epochs, lr, head_dim, val_split = get_hyperparams()

    wandb.init(
        entity="manikx",
        project="hallucination_circuits",
        config={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "epochs": epochs,
            "learning_rate": lr,
            "head_hidden_dim": head_dim,
            "val_split": val_split,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token  # ensure there's a pad token

    # 2) Load raw dataset (assumes each row has "query" and "sae_features")
    data_dir = os.getenv("DATA_DIR") or "/workspace/eli5_sae_features_parquet"
    dataset_id = os.getenv("HF_DATASET_ID", "mksethi/eli5_sae_features")
    ds = load_dataset("parquet", data_files=f"{data_dir}/*.parquet", split="train")

    # 3) Train/Validation split
    ds_split = ds.train_test_split(test_size=val_split, seed=42)
    train_ds, val_ds = ds_split["train"], ds_split["test"]

    # 4) Tokenize + format
    def tokenize_batch(examples):
        toks = tokenizer(
            examples["query"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "sae_features": examples["sae_features"],
        }

    # train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=ds.column_names)
    # val_ds   = val_ds.map(tokenize_batch, batched=True, remove_columns=ds.column_names)

    # train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "sae_features"])
    # val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "sae_features"])

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "sae_features"])
    val_ds.set_format(type="torch", columns=["input_ids","attention_mask","sae_features"])



    # 5) DataLoaders
    optimal_workers = min(num_workers, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=optimal_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # 6) Build model
    first_example = train_ds[0]
    sae_dim = len(first_example["sae_features"])
    model = Query2SAE(head_hidden_dim=head_dim, sae_dim=sae_dim).to(device)

    # 7) Optimizer & loss
    optimizer = optim.AdamW(model.head.parameters(), lr=lr)  # only train the head
    loss_fn   = nn.MSELoss()

    ckpt_dir = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # 8) Training loop
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, hit_count, n_seen = 0.0, 0, 0
        total_precision50, total_recall50, n_batches = 0.0, 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x_ids = batch["input_ids"].to(device)
            x_mask = batch["attention_mask"].to(device)
            y = batch["sae_features"].to(device).float()

            optimizer.zero_grad()
            preds = model(x_ids, attention_mask=x_mask)
            loss  = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred50 = preds.topk(50, dim=1).indices
            true50 = y.topk(50, dim=1).indices
            matches = (pred50.unsqueeze(2) == true50.unsqueeze(1)).sum(dim=(1,2))
            precision50 = (matches.float() / 50.0).mean().item()
            recall50 = precision50
            total_precision50 += precision50
            total_recall50 += recall50
            n_batches += 1

        avg_loss = total_loss / len(train_loader)
        avg_precision50 = total_precision50 / n_batches
        avg_recall50 = total_recall50 / n_batches

        wandb.log({
            "Train/Loss": avg_loss,
            "Train/Precision@50": avg_precision50,
            "Train/Recall@50": avg_recall50
        }, step=epoch)

        
        st_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.safetensors")
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(cpu_state, st_path)
        print(f"→ Saved SafeTensors checkpoint to {st_path}")

    # 9) Final eval
    val_loss, val_top50 = evaluate(model, val_loader, loss_fn, device)
    print(f"Validation | Loss: {val_loss:.4f} | Top-50: {val_top50:.4f}")
    wandb.log({"Val/Loss": val_loss, "Val/Top50": val_top50})

if __name__ == "__main__":
    main()
