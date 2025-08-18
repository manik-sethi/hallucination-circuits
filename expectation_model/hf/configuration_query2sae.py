from transformers import PretrainedConfig

class Query2SAEConfig(PretrainedConfig):
    model_type = "query2sae"

    def __init__(
        self,
        backbone_name: str = "gpt2",
        head_hidden_dim: int = 128,
        sae_dim: int = 1024,  # <-- set this to YOUR real SAE dim
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.head_hidden_dim = int(head_hidden_dim)
        self.sae_dim = int(sae_dim)
