from transformers import AutoModel, AutoTokenizer

MODEL_REGISTRY = {
    "gpt2-small": "gpt2",
    "gemma-2b-it": "google/gemma-2b-it",
    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct"
}

def load_model_and_tokenizer(name: str):
    hf_name = MODEL_REGISTRY.get(name)
    if not hf_name:
        raise ValueError(f"Unknown model: {name}")
    model = AutoModel.from_pretrained(hf_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return model, tokenizer
