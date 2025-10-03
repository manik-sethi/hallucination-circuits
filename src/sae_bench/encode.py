from sae_lens import SAE
from transformer_lens import HookedTransformer
import torch, yaml, os, shutil, glob, sys
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk
from functools import partial
import torch.distributed as dist


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from sae_bench.io import sae_codes_path, ensure_dir

# Distributed setup
local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
multi_gpu = world_size > 1

# Constants
BATCH_SIZE = 32
WRITER_BATCH_SIZE = 512
TEXT_COL = "Answer"
MAX_LEN = 1024
CONFIG_PATH = "configs/pipeline.yaml"
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def tokfn(batch, tokenizer, text_col, max_len):
    enc = tokenizer(
        batch[text_col], padding="max_length", truncation=True,
        max_length=max_len, return_attention_mask=True
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

@torch.inference_mode()
def token2sae(input_ids, attention_mask, halluc_batch, *, hooked_model, sae_model, hook_num, new_col_name):
    toks = torch.tensor(input_ids, device=device, dtype=torch.long)
    attn = torch.tensor(attention_mask, device=device, dtype=bool)
    hook_name = f"blocks.{hook_num}.hook_resid_post"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, cache = hooked_model.run_with_cache(
            toks, names_filter=lambda n: n == hook_name, stop_at_layer=hook_num+1
        )
        resid = cache[hook_name]
        reps = sae_model.encode(resid)
        mask = attn.unsqueeze(-1)
        reps_sum = (reps * mask).sum(dim=1)
        tok_counts = mask.sum(dim=1).clamp(min=1)
        acts = (reps_sum / tok_counts).float()
    out = acts.float().cpu().numpy()
    del cache, resid, reps, acts
    return {new_col_name: out, "Hallucination": halluc_batch}

def main():
    if multi_gpu:
        dist.init_process_group("nccl")

    cfg = load_config(CONFIG_PATH)
    model_keys = list(cfg.keys())

    for model_name in model_keys:
        model_config = cfg[model_name]
        layers = model_config['layers']
        model_id = model_config['hf_id']
        sae_release = model_config['sae_release']

        model = HookedTransformer.from_pretrained(
            model_id, dtype=torch.bfloat16, device=device
        ).eval()

        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

        print(f"Loading Dataset for {model_name} (rank {rank}/{world_size})")
        data = load_dataset("mksethi/HalQA")

        pretokenize = partial(tokfn, tokenizer=model.tokenizer, text_col=TEXT_COL, max_len=MAX_LEN)
        data = data.map(pretokenize, batched=True, batch_size=1024,
                        num_proc=max(1,(os.cpu_count() or 2)-1),
                        desc=f"Pretokenizing {model_name}")

        # Shard dataset per rank
        shard = {}
        for split_name, ds in data.items():
            shard[split_name] = ds.shard(num_shards=world_size, index=rank)
        shard = DatasetDict(shard)

        # Encode each layer
        for layer in layers:
            NEW_COL = "sae"
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=f"resid_post_layer_{layer}_trainer_1",
                device=device
            ).eval()

            sae_encoder_fn = partial(
                token2sae, hooked_model=model, sae_model=sae,
                hook_num=layer, new_col_name=NEW_COL
            )

            layer_data = shard.map(
                sae_encoder_fn, batched=True, batch_size=BATCH_SIZE,
                input_columns=["input_ids", "attention_mask", "Hallucination"],
                writer_batch_size=WRITER_BATCH_SIZE,
                load_from_cache_file=False,
                desc=f"SAE encoding L{layer} (rank {rank}/{world_size})"
            )

            # Save each shard locally with io.py
            save_path = sae_codes_path(model_name, layer, ext="disk") / f"rank{rank}"
            ensure_dir(save_path)
            layer_data.save_to_disk(str(save_path))

            del sae
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            sae_pattern = f"/workspace/.cache/huggingface/hub/models--*--*/snapshots/*/resid_post_layer_{layer}_*"
            for path in glob.glob(sae_pattern):
                shutil.rmtree(path, ignore_errors=True)

        # Only rank 0 merges
        if multi_gpu:
            dist.barrier()
            if rank == 0:
                merged_layers = {}
                for layer in layers:
                    shard_dirs = [sae_codes_path(model_name, layer, ext="disk") / f"rank{r}" for r in range(world_size)]
                    all_shards = [load_from_disk(str(d)) for d in shard_dirs]
                    merged_layers[f"layer_{layer}"] = concatenate_datasets([ds["train"] for ds in all_shards])
                DatasetDict(merged_layers).push_to_hub(f"mksethi/{model_name}_sae_reps",
                                                       commit_message=f"Added SAE activations for {model_name} all layers")
        else:
            merged_layers = {}
            for layer in layers:
                shard_dir = sae_codes_path(model_name, layer, ext="disk") / f"rank{rank}"
                merged_layers[f"layer_{layer}"] = load_from_disk(str(shard_dir))["train"]
            DatasetDict(merged_layers).push_to_hub(f"mksethi/{model_name}_sae_reps",
                                                   commit_message=f"Added SAE activations for {model_name} all layers")

        # Cleanup local artifacts
        for layer in layers:
            shutil.rmtree(sae_codes_path(model_name, layer, ext="disk"), ignore_errors=True)

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
