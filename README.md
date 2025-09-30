# Using Monosemantic Features From Sparse Auto-Encoders to Detect Hallucinations

[![Build Status](https://github.com/manik-sethi/hallucination-circuits/actions/workflows/ci.yml/badge.svg)](https://github.com/manik-sethi/hallucination-circuits/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Made with PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Weights & Biases](https://img.shields.io/badge/Weights%20&%20Biases-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)


## Overview

Current methods of detecting hallucinations depend on logit distribution, semantic self-consistency, or layer-wise propagation.
In our research, we explore whether it is possible to detect hallucinations using monosemantic features.

## Getting started

### Code and development environment

First, clone this repo on your local machine
```
git clone https://github.com/manik-sethi/hallucination-circuits
```
Next, run the following commands to install all dependencies in a virtual environment
```
python -m venv hallu-env
pip install -r requirements.txt
```

### Data
We use [domenicrosati/TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA) as our base dataset. We then run the following transformation on it
- Run the text through a model which has [pretrained SAE's](https://jbloomaus.github.io/SAELens/latest/sae_table/)
- Using the [Hooked Transformer](https://transformerlensorg.github.io/TransformerLens/index.html) version of the model, retrieve the residuals from a given layer
- Encode the residuals using the SAE

By doing this, we get a representation of our input text in a specific layer space using a specific sparse auto-encoder. Now we can build a dataset which contains these SAE representations, which will be useful for further analysis.

Since this process has already been completed, the dataset is publically available on [huggingface](https://huggingface.co/datasets/mksethi/sae-acts-llama31-8b-it)

### Logging and tracking experiments

We use [Weights & Biases](https://wandb.ai/site) to log and track our experiments.
If you're logged in, your default entity will be used (a fixed entity is not set in the config),
and you can set another entity with the `WANDB_ENTITY` environment variable.
Otherwise, the runs will be anonymous (you don't need to be logged in).


## Repository structure

Below, we give a description of the main files and directories in this repository.

```
hallucination-circuits/
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── data/               # Dataset loaders & preprocessors
│   ├── detectors/          # Model definitions & wrappers
│   ├── interfaces/         # Connectors to external tools like Neuronpedia or Goodfire SDK
│
├── expectation_model/      # Expectation model implementation
│   ├── __init__.py
│   ├── hf/                 # Defining custom architecture classes for hugging face
│   ├── models/             # Defining Frozen Language Model MLP class
│   ├── trainer/            # Training loop implementation
│   ├── utils/              # Supporting utils like model loading and metrics
│   └── ...                 # Miscellaneous files, can be treated like scratch work
│
├── notebooks/              # Jupyter notebooks for exploration
│   ├── layerwise_pc1_analysis.ipynb      # Analyzing change in variance explained by PC1 by layer
│   ├── pca1_magnitude_corr.ipynb         # Exploring correlation between first principal component and norm of data
│   └── truthfulqa_sae_encoding.ipynb     # Transforming truthfulqa text into sae representations and creating a dataset

```



## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.

The initial code of this repository has been initiated by the [Python Machine Learning Research Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
with the LICENSE.ml-template file.

Additional LICENSE files may be present in subdirectories of the project.
