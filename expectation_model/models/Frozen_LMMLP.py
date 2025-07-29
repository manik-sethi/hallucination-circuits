import torch
import torch.nn as nn

class FrozenLMMLP(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_dim: int, output_dim: int):
        super().__init__()
        self.base_model = base_model
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get hidden size from base_model config
        hidden_size = base_model.config.hidden_size

        # MLP head: project base_model hidden to SAE feature space
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Run base model to get hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state
        # Pool: mean over sequence dimension
        pooled = hidden_states.mean(dim=1)  # (batch_size, hidden_size)
        # Project to SAE feature dimension
        features = self.mlp(pooled)  # (batch_size, output_dim)
        return features
