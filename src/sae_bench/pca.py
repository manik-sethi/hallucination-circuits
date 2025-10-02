from sklearn.decomposition import PCA
from datasets import load_dataset
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import torch

from .io import pca_dir


repo_id = "mksethi/sae_reps_processed"

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_metrics(X: np.ndarray) -> np.ndarray:

  l1 = np.sum(np.abs(X), axis=1)
  l2 = np.linalg.norm(X, axis=1)

  pca = PCA(n_components=10)
  X_reduced = pca.fit_transform(X)
  pca.explained_variance_ratio_

  scores = X_reduced[:,0]

  df = pd.DataFrame({
      ## Create one for each model
      "PC1": scores,
      "l2-norm": l2,
      "l1-norm": l1,
    })

  return pca.explained_variance_ratio_, df



def main():

  cfg = load_config("configs/pipeline.yaml")
  model_keys = list(cfg.keys())
  for model_name in model_keys:

    layers = cfg[model_name]['layers']
    
    data = load_dataset(repo_id, split=model_name)

    columns = data.column_names

    dataset_numpy_view = data.with_format(
      type='numpy',
      columns=columns,
      output_all_columns=False
    )

  pc_ratio_dict = {}

  for column, layer in zip(columns, layers):
    X = dataset_numpy_view[column][:]
    pc_ratios, df = get_metrics(X)
    pc_ratio_dict[f'layer_{layer}_pc'] = pc_ratios

    layer_dir = pca_dir(model_name, layer)
    layer_file_path = layer_dir / f"{model_name}_layer_{layer}_norms.csv"
    df.to_csv(layer_file_path)

  model_path = pca_dir(model_name, layer).parent
  model_summary_file_path = model_path / "summary_pc_ratios.csv"

  pd.DataFrame(pc_ratio_dict).to_csv(model_summary_file_path)
  
    

if __name__ == "__main__":
  main()
