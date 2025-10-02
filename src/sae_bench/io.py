from __future__ import annotations
from pathlib import Path
import json

# Base dirs
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ART = ROOT / "artifacts"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# dataset
def combined_path() -> Path:
    ensure_dir(DATA)
    return DATA / "combined.arrow"

# SAE encoding
def sae_codes_path(model: str, layer: int, ext: str = "parquet") -> Path:
    p = ensure_dir(ART / "sae_codes" / model)
    return p / f"layer_{layer}.{ext}"

# PCA
def pca_dir(model: str, layer: int) -> Path:
    return ensure_dir(ART / "pca" / model / f"layer_{layer}")

# similarity
def sim_path(model_a: str, model_b: str, layer: int) -> Path:
    p = ensure_dir(ART / "similarity" / f"{model_a}__{model_b}")
    return p / f"layer_{layer}.parquet"

# regression
def reg_path(model: str, layer: int, space: str) -> Path:
    p = ensure_dir(ART / "regression" / space / model)
    return p / f"layer_{layer}.json"

# plot
def plots_dir() -> Path:
    return ensure_dir(ART / "plots")

# utils
def dump_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
