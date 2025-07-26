#src/data/huggingface.py
from datasets import load_dataset, Dataset as HFDataset
from transformer_lens.utils import tokenize_and_concatenate
from torch.utils.data import Dataset
from src.data.base import DatasetLoader
import torch
import pandas as pd

class TextDatasetLoader(Dataset, DatasetLoader):
    def __init__(
        self,
        # hf_link: str,
        list_str,
        tokenizer,
        sae,
    ):

        self.context_size = sae.cfg.context_size
        self.add_bos = sae.cfg.prepend_bos

        # raw_ds = load_dataset(hf_link, split=split, streaming=False)
        df = pd.DataFrame({"text": list_str})
        raw_ds = HFDataset.from_pandas(df)

        token_chunks = tokenize_and_concatenate(
            dataset=raw_ds,
            tokenizer=tokenizer,
            streaming=False,
            max_length=self.context_size,
            add_bos_token=self.add_bos
        )

        self.encodings = []
        for chunk in token_chunks:
            ids = chunk["tokens"]
            mask = torch.ones_like(ids, dtype=torch.long)
            self.encodings.append({
                "input_ids": ids,
                "attention_mask": mask
            })

        self.tokens = token_chunks

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx:int):
        return self.encodings[idx]