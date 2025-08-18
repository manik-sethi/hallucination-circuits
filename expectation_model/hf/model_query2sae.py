import torch
import torch.nn as nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model
from configuration_query2sae import Query2SAEConfig

class Query2SAEModel(PreTrainedModel):
    """
    HF-compatible wrapper for your Query2SAE:
    - GPT-2 backbone is frozen
    - MLP head maps hidden -> SAE space
    """
    config_class = Query2SAEConfig
    base_model_prefix = "query2sae"

    def __init__(self, config: Query2SAEConfig):
        super().__init__(config)
        # Build GPT-2 backbone (weights will be loaded by from_pretrained via state_dict)
        gpt2_cfg = GPT2Config.from_pretrained(config.backbone_name)
        self.backbone = GPT2Model(gpt2_cfg)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, config.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.head_hidden_dim, config.sae_dim),
        )

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = out.last_hidden_state[:, -1, :]
        logits = self.head(last_hidden)
        return {"logits": logits}
