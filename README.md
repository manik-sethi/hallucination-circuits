# Using Monosemantic Features From Sparse Auto-Encoders to Detect Hallucinations

[![PyPI version](https://img.shields.io/pypi/v/sae-lens)](https://pypi.org/project/sae-lens/)
[![Build Status](https://github.com/manik-sethi/hallucination-circuits/actions/workflows/ci.yml/badge.svg)](https://github.com/manik-sethi/hallucination-circuits/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Made with PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Weights & Biases](https://img.shields.io/badge/Weights%20&%20Biases-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)


## Overview

> [!NOTE]
> **TEMPLATE TODO:**
> Replace the description below with a description of your project, then delete this note.

A template for starting Python machine-learning research
projects with hardware acceleration featuring:

Current methods of detecting hallucinations depend on logit distribution, semantic self-consistency, or layer-wise propagation.
In our research, we explore whether it is possible to detect hallucinations using monosemantic features.

- ✅ Reproducible environments on major platforms with hardware acceleration with a great development experience
  covering multiple use cases:
    - 💻 local machines, e.g., macOS (+ Apple Silicon/MPS) and Linux/Windows WSL (+ NVIDIA GPU).
    - 🌐 Remote Linux servers with GPUs, e.g., VMs on cloud providers and IC and RCP HaaS at EPFL.
    - ☁️ Managed clusters supporting OCI containers with GPUs, e.g., the EPFL IC and RCP Run:ai (Kubernetes) clusters
      and the SCITAS Slurm clusters.
- 📦 Python project packaging following the
  [PyPA packaging guidelines](https://packaging.python.org/en/latest/tutorials/packaging-projects/) to avoid hacky
  imports.
- 📊 Experiment management, tracking, and sharing with [Hydra](https://hydra.cc/)
  and [Weights & Biases](https://wandb.ai/site).
- 💾 Checkpointing setup for research experiments compatible with Weights & Biases.
- 🧹 Code quality with [pre-commit](https://pre-commit.com) hooks.

🤝 The template makes collaboration and open-sourcing straightforward, avoiding setup issues and
[maximizing impact](https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501#a826).

🏆 The practices in this template earned its authors
an [Outstanding Paper (Honorable Mention)](https://openreview.net/forum?id=E0qO5dI5aEn)
at the [ML Reproducibility Challenge 2022](https://paperswithcode.com/rc2022).

📌 Projects made with the template would look like
[this toy project](https://github.com/skandermoalla/pytoych-benchmark)
or [this paper](https://github.com/CLAIRE-Labo/no-representation-no-trust) whose curves have been exactly reproduced
(exact same numbers) on multiple different platforms (EPFL Kubernetes cluster, VM on GCP, HPC cluster with Apptainer).

📖 Follow this README to get started with the template.

For a brief discussion of the template's design choices, features, and a Q&A check `template/README.md` file.

## Getting started

### Code and development environment

Once you have cloned this repo on your local machine, first install the dependencies by running the following code.
```
pip install -r requirements.txt
```

> [!NOTE]
> **TEMPLATE TODO**:
> Update the installation methods and platforms you support, delete the rest, and delete this note.
> I.e. keep either Docker or Conda, or both, or multiple of each if you support multiple platforms.
> 1. Specify the platform for each option and its description
>    e.g., for Docker amd64, arm64, etc., and for conda osx-arm64, linux-amd64, etc.
> 2. Specify the hardware acceleration options for each platform
>    e.g., for Docker NVIDIA GPUs, AMD GPUs etc.
> 3. Specify the hardware on which you ran your experiments (e.g., type of CPU/GPU and size of memory) and
>    the minimum hardware required to run your code if applicable (e.g., NVIDIA GPU with 80GB of memory).

We support the following methods and platforms for installing the project dependencies and running the code.

- **Docker/OCI-container for AMD64 machines (+ NVIDIA GPUs)**:
  This option works for machines with AMD64 CPUs and NVIDIA GPUs.
  E.g. Linux machines (EPFL HaaS servers, VMs on cloud providers),
  Windows machines with WSL, and clusters running OCI-compliant containers,
  like the EPFL Run:ai (Kubernetes) clusters.

  Follow the instructions in `installation/docker-amd64-cuda/README.md` to install the environment
  then get back here for the rest of the instructions to run the experiments.

  We ran our experiments on TODO: FILL IN THE HARDWARE YOU USED.
  To run them, you should have at least TODO: FILL IN THE MINIMUM HARDWARE REQS IF APPLICABLE.

- **Conda for osx-arm64**
  This option works for macOS machines with Apple Silicon and can leverage MPS acceleration.

  Follow the instructions in `installation/conda-osx-arm64-mps/README.md` to install the environment
  then get back here for the rest of the instructions to run the experiments.

  We ran our experiments on TODO: FILL IN THE HARDWARE YOU USED.
  To run them, you should have at least TODO: FILL IN THE MINIMUM HARDWARE REQS IF APPLICABLE.

### Data

> [!NOTE]
> **TEMPLATE TODO**:
> Fill `data/README.md` or delete this section, then delete this note.

Refer to `data/README.md`.

### Logging and tracking experiments

We use [Weights & Biases](https://wandb.ai/site) to log and track our experiments.
If you're logged in, your default entity will be used (a fixed entity is not set in the config),
and you can set another entity with the `WANDB_ENTITY` environment variable.
Otherwise, the runs will be anonymous (you don't need to be logged in).

## Reproduction and experimentation

### Reproducing our results

> [!NOTE]
> **TEMPLATE TODO**:
> Keep these scripts up to date and run your experiments using them.
> Do provide the W&B runs and trained models or update this section.
> Delete this note when shipping.

We provide scripts to reproduce our work in the `reproducibility-scripts/` directory.
It has a README at its root describing which scripts reproduce which experiments.

We share our Weights and Biases runs in [this W&B project](https://wandb.ai/claire-labo/template-project-name).

Moreover, we make our trained models available.
You can follow the instructions in `outputs/README.md` to download and use them.

### Experiment with different configurations

The default configuration for each script is stored in the `configs/` directory.
They are managed by [Hydra](https://hydra.cc/docs/intro/).
You can experiment with different configurations by passing the relevant arguments.
You can get examples of how to do so in the `reproducibility-scripts/` directory.

## Repository structure

> [!NOTE]
> **TEMPLATE TODO**:
> Provide a quick overview of the main files in the repo for users to understand your code,
> then delete this note.

Below, we give a description of the main files and directories in this repository.

```
 └─── src/                              # Source code.
    └── template_package_name           # Our package.
        ├── configs/                    # Hydra configuration files.
        └── template_experiment.py      # A template experiment.
```

## Contributing

We use [`pre-commit`](https://pre-commit.com) hooks to ensure high-quality code.
Make sure it's installed on the system where you're developing
(it is in the dependencies of the project, but you may be editing the code from outside the development environment.
If you have conda you can install it in your base environment, otherwise, you can install it with `brew`).
Install the pre-commit hooks with

```bash
# When in the PROJECT_ROOT.
pre-commit install --install-hooks
```

Then every time you commit, the pre-commit hooks will be triggered.
You can also trigger them manually with:

```bash
pre-commit run --all-files
```

## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.

The initial code of this repository has been initiated by the [Python Machine Learning Research Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
with the LICENSE.ml-template file.

Additional LICENSE files may be present in subdirectories of the project.
