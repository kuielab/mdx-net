# KUIELab-MDX-Net

## 0. Environment

- Ubuntu 20.04
- at least four cuda-able GPUs (each > 2080ti)
- 1.5 TB disk storage for data augmentation

## 1. Installation

```bash
conda env create -f conda_env_gpu.yaml -n mdx-net
conda activate mdx-net
pip install -r requirements.txt

sudo apt-get install soundstretch
```

## 2. Training & Submission

see [README_SUBMISSION.md](README_SUBMISSION.md)

# ACKNOWLEDGEMENT

- This repository is based on [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template)
- Also, facebook/[demucs](https://github.com/facebookresearch/demucs)