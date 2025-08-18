import torch
import torch.nn as nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model
from .my_configuration import Query2SAEConfig

class Query2SAEModel(PreTrainedModel):
    """
    Hugging Face-compatible wrapper around your Query2SAE.
    - Freezes the GPT-2 backbone
    - Adds a small MLP head to predict SAE features
    - Saves/loads with save_pretrained()/from_pretrained()
    """
    config_class = Query2SAEConfig
    base_model_prefix = "query2sae"

    def __init__(self, config: Query2SAEConfig):
        super().__init__(config)

        # Build GPT-2 backbone WITHOUT downloading weights (weights are loaded by from_pretrained)
        gpt2_cfg = GPT2Config.from_pretrained(config.backbone_name)
        self.backbone = GPT2Model(gpt2_cfg)

        # Freeze backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Head maps hidden_size -> head_hidden_dim -> sae_dim
        self.head = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, config.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.head_hidden_dim, config.sae_dim),
        )

        # Initialize head weights the HF way
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # no grad through backbone (keeps it frozen and faster)
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = out.last_hidden_state[:, -1, :]  # [B, hidden_size]
        logits = self.head(last_hidden)                   # [B, sae_dim]
        return {"logits": logits, "last_hidden_state": out.last_hidden_state}

    # Optional helpers for HF-style naming consistency
    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.backbone.set_input_embeddings(value)
