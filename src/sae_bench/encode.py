from sae_lens import SAE
from transformer_lens import HookedTransformer
import torch
import yaml
import sys
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_config(config_path: str):
    with open("configs/pipeline.yaml", "r") as f:
        return yaml.safe_load(f)
        
@torch.inference_mode()
def token2sae(texts, model, sae):                    
    toks = model.to_tokens(texts)
    _, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: n == layer,  # cache only this hook
        stop_at_layer=i + 1,                # stop early
    )
    resid = cache[layer]                    # [B, L, d_model]
    reps  = sae.encode(resid)               # [B, L, d_sae]
    acts  = reps.mean(dim=1)                # [B, d_sae]
    out   = acts.float().cpu().numpy().tolist()

    del cache, resid, reps, acts
    
    return {NEW_COL: out}


def main():
    cfg = load_config("configs/pipeline.yaml")
    layers = cfg['llama-3.1-8b-Instruct']['layers']
    model_id = cfg['llama-3.1-8b-Instruct']['hf_id']

    model = HookedTransformer.from_pretrained(model_id)
    print("Loading Dataset")
    data = load_dataset("mksethi/HalQA")
    
    for layer in layers:
        
        sae = SAE.from_pretrained(
            release = cfg['llama-3.1-8b-Instruct']['sae_release'],
            sae_id = f"resid_post_layer_{layer}_trainer_1",
            device = device
        ).eval()

        extra_args = {
            "model": model,
            "sae": sae
        }

        sae_layer_rep = data.map(
            token2sae,
            batched=True,
            batch_size=16,
            input_columns=["Answer", "Hallucination"],
            writer_batch_size=32,
            load_from_cache_file=False,
            fn_kwargs=extra_args
        )
        
    del sae

if __name__ == "__main__":
    main()

