from sae_lens import SAE
from transformer_lens import HookedTransformer
import torch
import yaml
import sys
from datasets import load_dataset, DatasetDict, Value, Sequence

from uuid import uuid4 

BATCH_SIZE = 16
TEXT_COL = "Answer"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
        
@torch.inference_mode()
def token2sae(batch, model, sae, hook_num, new_col_name):
    
    toks = model.to_tokens(batch[TEXT_COL])

    hook_name = f"blocks.{hook_num}.hook_resid_post"
    
    _, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: n == hook_name,  # cache only this hook
        stop_at_layer=hook_num + 1,                # stop early
    )
    resid = cache[hook_name]                    # [B, L, d_model]
    reps  = sae.encode(resid)               # [B, L, d_sae]
    acts  = reps.mean(dim=1)                # [B, d_sae]
    out   = acts.float().cpu().numpy().tolist()

    del cache, resid, reps, acts
    
    return {new_col_name: out,
           "Hallucination": batch["Hallucination"]
           }


def main():
    cfg = load_config("configs/pipeline.yaml")

    
    model_keys = list(cfg.keys())
    repo_id = "mksethi/sae_reps_processed"
    
    for model_name in model_keys:
        model_config = cfg[model_name]
        layers = model_config['layers']
        model_id = model_config['hf_id']
        sae_release = model_config['sae_release']

        # Load the hooked model from transformerlens
        model = HookedTransformer.from_pretrained(model_id)
        print(f"Loading Dataset for {model_name}")
        data = load_dataset("mksethi/HalQA")
        
        for layer in layers:
            
            NEW_COL = f"{model_name}_sae_l{layer}"
            
            sae = SAE.from_pretrained(
                release = sae_release,
                sae_id = f"resid_post_layer_{layer}_trainer_1",
                device = device
            ).eval()
    
            extra_args = {
                "model": model,
                "sae": sae,
                "new_col_name": NEW_COL,
                "hook_num": layer
            }

            new_splits = {}

            for name, split in data.items():
                feats = split.features.copy()

                feats[NEW_COL] = Sequence(Value("float32"))

                input_cols = [TEXT_COL, "Hallucination"]
    
                new_splits[name] = split.map(
                    token2sae,
                    batched=True,
                    batch_size=BATCH_SIZE,
                    input_columns=input_cols,
                    features=feats,
                    writer_batch_size=32,
                    load_from_cache_file=False,
                    desc=f"SAE encoding L{layer}",
                    new_fingerprint=str(uuid4()),
                    fn_kwargs=extra_args
                )
            data = DatasetDict(new_splits)
    
            del sae

            if device == 'cuda':
                torch.cuda.empty_cache()
            
        data.push_to_hub(repo_id,
                         split=model_name,
                        commit_message=f"Added SAE activations for {model_name}")
            
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

