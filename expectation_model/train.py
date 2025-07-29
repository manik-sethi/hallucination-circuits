import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import argparse
from models.Frozen_LMMLP import FrozenLMMLP
from sae_lens import SAE

def calculate_top_k_accuracy(predictions, targets, k=50):
    """Calculate top-k accuracy for SAE feature prediction"""
    batch_size = predictions.size(0)
    
    # Get top-k predicted indices
    _, pred_indices = torch.topk(predictions, k, dim=1)
    
    # Get top-k target indices  
    _, target_indices = torch.topk(targets, k, dim=1)
    
    # Calculate intersection
    correct = 0
    for i in range(batch_size):
        pred_set = set(pred_indices[i].cpu().numpy())
        target_set = set(target_indices[i].cpu().numpy())
        intersection = len(pred_set.intersection(target_set))
        correct += intersection / k  # Normalized by k
    
    return correct / batch_size

def calculate_metrics(predictions, targets):
    """Calculate various accuracy metrics"""
    metrics = {}
    
    # Top-k accuracies
    for k in [10, 50, 100, 200]:
        metrics[f'top_{k}_accuracy'] = calculate_top_k_accuracy(predictions, targets, k)
    
    # Spearman correlation (measure of ranking similarity)
    batch_size = predictions.size(0)
    correlations = []
    for i in range(batch_size):
        pred_ranks = torch.argsort(torch.argsort(predictions[i], descending=True))
        target_ranks = torch.argsort(torch.argsort(targets[i], descending=True))
        
        # Calculate Spearman correlation
        pred_ranks_f = pred_ranks.float()
        target_ranks_f = target_ranks.float()
        
        pred_mean = pred_ranks_f.mean()
        target_mean = target_ranks_f.mean()
        
        numerator = ((pred_ranks_f - pred_mean) * (target_ranks_f - target_mean)).sum()
        pred_std = ((pred_ranks_f - pred_mean) ** 2).sum().sqrt()
        target_std = ((target_ranks_f - target_mean) ** 2).sum().sqrt()
        
        if pred_std > 0 and target_std > 0:
            correlation = numerator / (pred_std * target_std)
            correlations.append(correlation.item())
    
    if correlations:
        metrics['spearman_correlation'] = sum(correlations) / len(correlations)
    
    # L2 distance between normalized vectors
    pred_norm = F.normalize(predictions, p=2, dim=1)
    target_norm = F.normalize(targets, p=2, dim=1)
    metrics['cosine_similarity'] = F.cosine_similarity(pred_norm, target_norm).mean().item()
    
    return metrics

# Fix wandb initialization - remove entity or use correct one
wandb.init(project="hallucination_circuits")  # Removed entity parameter

def get_dataloaders(tokenizer, sae, base_model, context_size, batch_size):
    dataset = load_dataset("kilt_tasks", "eli5")
    device = next(base_model.parameters()).device

    def preprocess(example):
        # Fix: ELI5 dataset has different structure - use the answer text
        # The output field contains a list of answers, take the first one
        if isinstance(example["output"], list) and len(example["output"]) > 0:
            text = example["output"][0]["answer"]
        else:
            # Fallback to input if output format is unexpected
            text = str(example.get("input", ""))
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=context_size,
            return_tensors="pt"
        )
        
        # Get hidden states from base model first, then pass to SAE
        with torch.no_grad():
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Get hidden states from the base model
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # Shape: [1, seq_len, hidden_size]
            
            # Use mean pooling to match the model's forward pass
            pooled_hidden = hidden_states.mean(dim=1)  # Shape: [1, hidden_size]
            
            # Now pass the pooled hidden states to SAE
            sae_out = sae.encode(pooled_hidden)  # Should work now
            if len(sae_out.shape) > 1:
                sae_out = sae_out.squeeze(0)  # Remove batch dimension

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": sae_out.detach().cpu()
        }

    # Take a smaller subset for testing
    train_dataset = dataset["train"].select(range(100))  # Use first 1000 examples
    tokenized_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    tokenized_dataset.set_format(type='torch')

    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sae, _, _ = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=device
    )
    base_model = AutoModel.from_pretrained(args.base_model)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Move base_model to device before using it in preprocessing
    base_model = base_model.to(device)
    
    # Debug SAE dimensions
    print(f"SAE W_enc shape: {sae.W_enc.shape}")
    print(f"SAE d_sae: {sae.cfg.d_sae}")
    
    # Use the correct SAE feature dimension
    sae_feature_dim = sae.cfg.d_sae  # This should be 24576
    
    # Fixed: pass base_model object, not args.base_model string
    model = FrozenLMMLP(base_model, hidden_dim=args.hidden_dim, output_dim=sae_feature_dim).to(device)
    train_loader = get_dataloaders(tokenizer, sae, base_model, args.context_size, args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.mse_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Store predictions and targets for metrics calculation
            all_predictions.append(logits.detach())
            all_targets.append(labels.detach())

        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_predictions, all_targets)
        avg_loss = total_loss / len(train_loader)
        
        # Log to wandb
        log_dict = {"epoch": epoch + 1, "loss": avg_loss}
        log_dict.update(metrics)
        wandb.log(log_dict)
        
        # Print metrics
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        print(f"Top-50 Accuracy: {metrics['top_50_accuracy']:.3f}")
        print(f"Top-100 Accuracy: {metrics['top_100_accuracy']:.3f}")
        print(f"Cosine Similarity: {metrics['cosine_similarity']:.3f}")
        print(f"Spearman Correlation: {metrics.get('spearman_correlation', 0):.3f}")
        print("-" * 50)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="gpt2")
    parser.add_argument("--context_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sae_release", type=str, default="gpt2-small-res-jb")
    parser.add_argument("--sae_id", type=str, default="blocks.8.hook_resid_pre")
    args = parser.parse_args()

    train(args)