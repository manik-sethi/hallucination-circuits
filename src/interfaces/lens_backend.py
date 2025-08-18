#src/interfaces/lens_backend.py
import warnings
from sae_lens import SAE, SAEConfig, HookedSAETransformer
from transformer_lens import HookedTransformer
import torch

class Variant:
    def __init__(self, model_id, sae_release, sae_id):
        self.model_id = model_id
        self.sae_release = sae_release
        self.sae_id = sae_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # Loading in SAE
            self.sae, cfg_dict, _ = SAE.from_pretrained(
                release=self.sae_release,
                sae_id=self.sae_id,
                device=self.device
            )

        self.cfg = SAEConfig.from_dict(cfg_dict)
        
        # Loading in the model that was used to train SAE
        kwargs = self.cfg.model_from_pretrained_kwargs or {}

        if kwargs:  # SAE-specific config exists
            print("using HookedSAETransformer")
            self.model = HookedSAETransformer.from_pretrained_no_processing(
                model_id, device=self.device, **kwargs
            )
        else:  # Standard case
            print("using HookedTransformer")
            from transformer_lens import HookedTransformer
            self.model = HookedTransformer.from_pretrained(model_id, device=self.device)
        self.tokenizer = self.model.tokenizer
    
    def get_components(self):
        return self.model, self.sae, self.cfg, self.tokenizer
