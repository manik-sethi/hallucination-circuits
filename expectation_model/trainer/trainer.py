# expectation_model/trainer/trainer.py
import os
import torch
import wandb
from tqdm import tqdm
from utils.metrics import compute_mse, compute_cosine_similarity
from datasets import load_dataset
from torch.utils.data import DataLoader
from sae_lens import SAE
from models.Frozen_LMMLP import FrozenLMPLP as FrozenLMMLP


def build_dataloader(tokenizer, context_size, batch_size, sae, base_model):
    dataset = load_dataset("kilt_tasks", "eli5")

    def preprocess(example):
        if isinstance(example["output"], list) and len(example["output"]) > 0:
            text = example["output"][0]["answer"]
        else:
            text = str(example.get("input", ""))
        
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=context_size,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            # No .to(device) here - everything stays on CPU during preprocessing
            outputs = base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            pooled = outputs.last_hidden_state.mean(dim=1)
            sae_out = sae.encode(pooled).squeeze(0)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": sae_out
        }

    # Process sequentially without multiprocessing
    tokenized = dataset["train"].map(
        preprocess,
        remove_columns=dataset["train"].column_names,
        num_proc=1  # Force single process
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing in DataLoader
        pin_memory=True
    )


class Trainer:
    def __init__(self, model, train_loader, sae, device, args):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.sae = sae
        self.device = device
        self.args = args
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

    def train(self):
        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            total_loss = 0.0
            total_cosine = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                preds = self.model(input_ids=inputs, attention_mask=masks)
                loss = torch.nn.functional.mse_loss(preds, labels)
                cosine = compute_cosine_similarity(preds, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_cosine += cosine

            avg_loss = total_loss / len(self.train_loader)
            avg_cosine = total_cosine / len(self.train_loader)
            wandb.log({"epoch": epoch, "mse_loss": avg_loss, "cosine_similarity": avg_cosine})
            print(f"Epoch {epoch} | MSE Loss: {avg_loss:.4f} | CosineSim: {avg_cosine:.4f}")