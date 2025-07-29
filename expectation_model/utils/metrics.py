import torch

def compute_mse(preds, targets):
    return torch.nn.functional.mse_loss(preds, targets).item()

def compute_cosine_similarity(preds, targets):
    cos = torch.nn.functional.cosine_similarity(preds, targets, dim=-1)
    return cos.mean().item()
