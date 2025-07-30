#!/usr/bin/env python3
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, load_dataset_builder
from dotenv import load_dotenv
from tqdm import tqdm
import wandb

def get_hyperparams():
    batch_size  = int(os.getenv("BATCH_SIZE", 2048))
    num_workers = int(os.getenv("NUM_WORKERS", 4))
    epochs      = int(os.getenv("EPOCHS", 10))
    lr          = float(os.getenv("LEARNING_RATE", 1e-4))
    hidden_dim  = int(os.getenv("HIDDEN_DIM", 128))
    val_split   = float(os.getenv("VAL_SPLIT", 0.1))
    return batch_size, num_workers, epochs, lr, hidden_dim, val_split

class FrozenLMMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    hit_count  = 0
    n_seen     = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            x = batch[input_key].to(device, non_blocking=True).float()
            y = batch[label_key].to(device, non_blocking=True).float()
            preds = model(x)
            loss = loss_fn(preds, y)
            total_loss += loss.item()
            pred50 = preds.topk(50, dim=1).indices
            true50 = y.topk(50, dim=1).indices
            hits   = (pred50.unsqueeze(2) == true50.unsqueeze(1)).any(dim=(1,2))
            hit_count += hits.sum().item()
            n_seen    += hits.size(0)
    avg_loss = total_loss / len(loader)
    top50    = hit_count / n_seen
    return avg_loss, top50


def main():
    load_dotenv()
    batch_size, num_workers, epochs, lr, hidden_dim, val_split = get_hyperparams()

    wandb.init(
        entity="manikx",
        project="hallucination_circuits",
        config={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "epochs": epochs,
            "learning_rate": lr,
            "hidden_dim": hidden_dim,
            "val_split": val_split,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"- CUDA devices: {torch.cuda.device_count()}")
        print(f"- Current GPU name: {torch.cuda.get_device_name(0)}")

    data_dir = os.getenv("DATA_DIR") or "/workspace/eli5_sae_features_parquet"
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")

    dataset_id = os.getenv("HF_DATASET_ID", "mksethi/eli5_sae_features")
    builder = load_dataset_builder(dataset_id)
    features = builder.info.features

    ds = load_dataset(
        "parquet",
        data_files=f"{data_dir}/*.parquet",
        features=features,
        split="train"
    )

    first_py  = ds[0]
    global input_key, label_key
    input_key = "input_ids" if "input_ids" in first_py else next(k for k,v in first_py.items() if isinstance(v, list))
    label_key = "sae_features"

    # Split dataset
    ds_split = ds.train_test_split(test_size=val_split, seed=42)
    train_ds = ds_split['train']
    val_ds   = ds_split['test']

    print(f"Dataset sizes - Train: {len(train_ds)}, Validation: {len(val_ds)}")

    # Format as Tensors only
    train_ds.set_format(type="torch", columns=[input_key, label_key])
    val_ds.set_format(type="torch", columns=[input_key, label_key])

    # DataLoaders
    optimal_workers = min(num_workers, os.cpu_count() or 4)
    print(f"Using {optimal_workers} data loader workers")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    # Infer dimensions
    first_batch = next(iter(train_loader))
    input_dim  = first_batch[input_key].shape[1]
    output_dim = first_batch[label_key].shape[1]
    print(f"Detected input_dim={input_dim}, output_dim={output_dim}")

    # Model, optimizer, loss
    model    = FrozenLMMLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    # wandb.watch(model, log="all", log_freq=100)

    # Checkpoint dir
    ckpt_dir = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    epoch_losses=[]
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        hit_count  = 0
        n_seen     = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x = batch[input_key].to(device, non_blocking=True).float()
            y = batch[label_key].to(device, non_blocking=True).float()

            optimizer.zero_grad()
            preds = model(x)
            loss  = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred50 = preds.topk(50, dim=1).indices
            true50 = y.topk(50, dim=1).indices
            hits   = (pred50.unsqueeze(2) == true50.unsqueeze(1)).any(dim=(1,2))
            hit_count += hits.sum().item()
            n_seen    += hits.size(0)

        top50    = hit_count / n_seen
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        wandb.log({
            "Train/Epoch": epoch,
            "Train/Loss": avg_loss,
            "Train/Top 50 Overlap": top50,
        }, step=epoch)

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Top‑50 overlap: {top50:.4f}")

        ckpt_path = os.path.join(ckpt_dir, f"frozen_lm_mlp_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(ckpt_path)

    # Final evaluation on validation set
    print("Running final validation...")
    val_loss, val_top50 = evaluate(model, val_loader, loss_fn, device)
    print(f"Validation Loss: {val_loss:.4f} | Top‑50 overlap: {val_top50:.4f}")
    wandb.log({
        "Validation/loss": val_loss,
        "Validation/Top 50 Overlap": val_top50
    })

    model_artifact = wandb.Artifact(
        name="frozen-lm-mlp-model",
        type="model",
        description="FrozenLMMLP trained on SAE features",
    )
    model_artifact.add_file(ckpt_path)
    wandb.log_artifact(model_artifact)
    # Cleanup
    if os.getenv("REMOVE_CHECKPOINTS_AFTER_RUN", "false").lower() == "true":
        shutil.rmtree(ckpt_dir)

if __name__ == "__main__":
    main()
