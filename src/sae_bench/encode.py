from sae_lens import SAE
from transformer_lens import HookedTransformer
import torch
import yaml
import sys
import os
from datasets import load_dataset, DatasetDict, Value, Sequence
from functools import partial
from uuid import uuid4 
import shutil, glob

BATCH_SIZE = 32
WRITER_BATCH_SIZE = 512
TEXT_COL = "Answer"
MAX_LEN = 1024                # cap sequence length based on your VRAM
CONFIG_PATH = "configs/pipeline.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def tokfn(batch, tokenizer, text_col, max_len):
    enc = tokenizer(
        batch[text_col],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_attention_mask=True
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"]
    }
    
@torch.inference_mode()
def token2sae(input_ids, attention_mask, halluc_batch, *, hooked_model, sae_model, hook_num, new_col_name):
    
    toks = torch.tensor(input_ids, device=device, dtype=torch.long)
    attn = torch.tensor(attention_mask, device=device, dtype=bool)
    
    
    hook_name = f"blocks.{hook_num}.hook_resid_post"

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, cache = hooked_model.run_with_cache(
            toks,
            names_filter=lambda n: n == hook_name,
            stop_at_layer=hook_num + 1,
        )
        resid = cache[hook_name]        # [B, L, d_model]
        reps  = sae_model.encode(resid) # [B, L, d_sae]

        mask = attn.unsqueeze(-1)
        reps_sum = (reps * mask).sum(dim=1)
        tok_counts = mask.sum(dim=1).clamp(min=1)
        acts = (reps_sum / tok_counts).float()

    out   = acts.float().cpu().numpy()

    del cache, resid, reps, acts

    return {new_col_name: out, "Hallucination": halluc_batch}


def main():
    cfg = load_config(CONFIG_PATH)
    model_keys = list(cfg.keys())

    
    for model_name in model_keys:
        model_config = cfg[model_name]
        layers = model_config['layers']
        model_id = model_config['hf_id']
        sae_release = model_config['sae_release']

        # Load the hooked model from transformerlens
        model = HookedTransformer.from_pretrained(
            model_id, 
            dtype=torch.bfloat16, 
            device=device
        ).eval()

        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

        
        print(f"Loading Dataset for {model_name}")
        data = load_dataset("mksethi/HalQA")

        pretokenize = partial(
            tokfn,
            tokenizer=model.tokenizer,
            text_col=TEXT_COL,
            max_len=MAX_LEN
        )

        data = data.map(
            pretokenize,
            batched=True,
            batch_size=1024,
            num_proc=max(1,(os.cpu_count() or 2) - 1),
            desc = f"Pretokenizing {model_name}"
        )
        layer_dict = {}
        for layer in layers:
            
            NEW_COL = f"sae_l{layer}"
            
            sae = SAE.from_pretrained(
                release = sae_release,
                sae_id = f"resid_post_layer_{layer}_trainer_1",
                device = device
            ).eval()

            sae_encoder_fn = partial(
                token2sae,
                hooked_model=model,
                sae_model=sae,
                hook_num=layer,
                new_col_name=NEW_COL
            )

            layer_data = data.map(
                sae_encoder_fn,
                batched=True,
                batch_size=BATCH_SIZE,
                input_columns=["input_ids", "attention_mask", "Hallucination"],
                writer_batch_size=WRITER_BATCH_SIZE,
                load_from_cache_file=False,
                desc=f"SAE encoding L{layer}",
            )
            
            layer_dict[f"layer_{layer}"] = layer_data["train"]

            del sae
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            sae_pattern = f"/workspace/.cache/huggingface/hub/models--*--*/snapshots/*/resid_post_layer_{layer}_*"
            for path in glob.glob(sae_pattern):
                shutil.rmtree(path, ignore_errors=True)
            print(f"🧹 Cleared SAE cache for layer {layer}")
            
        REPO_ID = f"mksethi/{model_name}_sae_reps"

        DatasetDict(layer_dict).push_to_hub(
            REPO_ID,
            commit_message=f"Added SAE activations for {model_name} all layers"
        )
        
        try:
            # Clean all HF datasets cache to reclaim space
            shutil.rmtree("/workspace/.cache/huggingface/datasets", ignore_errors=True)
        except Exception:
            pass


                            

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

        for path in glob.glob(f"/workspace/.cache/huggingface/hub/models--*{model_id.replace('/', '--')}*"):
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()

