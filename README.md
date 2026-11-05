# CVSG: Coalition Value and Synergy-Guided Multimodal Fusion for Sentiment Analysis

CVSG is a multimodal sentiment analysis framework for continuous sentiment prediction from language, acoustic, and visual signals. It learns modality allocation and pairwise interaction strength using prediction quality measured over modality coalitions.

## Overview

CVSG consists of three modules:

1. **Distributional Modality Encoding**: encodes language, acoustic, and visual sequences and constructs modality-specific Gaussian representations. Pooled modality features and posterior-scale descriptors are passed to the fusion module.
2. **Value and Synergy Fusion**: predicts modality utilities for content allocation and pairwise gains for interaction correction. The fused representation is processed by a Gaussian bottleneck and a sentiment prediction head.
3. **Coalition Value Supervision**: during training, evaluates all seven nonempty modality subsets. Their supervised prediction losses define marginal-utility and pair-gain targets for the fusion predictors. This auxiliary branch is not needed for ordinary inference.

## Project structure

```text
CVSG/
├── main.py                      # Training, validation, testing, and checkpoints
├── model.py                     # CVSG model and learning objectives
├── global_configs.py           # Feature dimensions and device defaults
├── requirements.txt            # Python dependencies
├── modules/
│   ├── __init__.py
│   ├── transformer.py
│   ├── multihead_attention.py
│   └── position_embedding.py
├── scripts/
│   └── check_dataset.py
├── datasets/
│   ├── README.md
│   ├── mosi.pkl                 # Download separately; not included
│   └── mosei.pkl                # Download separately; not included
├── bert-base-uncased/           # Download separately; not included
├── checkpoints/                # Created by training
└── results/                    # Training and evaluation outputs
```

## 1. Environment setup

Python 3.10 or 3.11 is recommended. For GPU training, first install a PyTorch build compatible with the installed CUDA driver, following the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Install the remaining requirements from the repository root:

```bash
python -m pip install -r requirements.txt
```

## 2. Download the CMU-MOSI and CMU-MOSEI datasets

CVSG expects **word-aligned** multimodal examples in Python `.pkl` files. The supported data format and a public download script are available at:

- Dataset download script: https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/master/datasets/download_datasets.sh
- CMU-MOSI dataset information: https://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/
- CMU-MOSEI dataset information: https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/
- CMU Multimodal Data SDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK

### Option A: Download preprocessed `.pkl` files

Run these commands from the **CVSG repository root**:

```bash
python -m pip install gdown
mkdir -p datasets
gdown "https://drive.google.com/uc?id=12HbavGOtoVCqicvSYWl3zImli5Jz0Nou" -O datasets/mosi.pkl
gdown "https://drive.google.com/uc?id=1VJhSc2TGrPU8zJSVTYwn5kfuG47VaNQ3" -O datasets/mosei.pkl
```

The Google Drive IDs above come from the public dataset download script. If an ID is unavailable or its access permissions change, use the script's alternate download links. A raw CMU SDK download is **not** directly interchangeable with the required preprocessed `.pkl` format; additional preprocessing and matching train/validation/test splits would be necessary.

On Windows PowerShell, create the folder with `New-Item -ItemType Directory -Force datasets` instead of `mkdir -p datasets`.

### Expected data structure

The loader expects a dictionary of splits, where each sample is:

```python
((words, visual_features, acoustic_features), sentiment_score, sample_id)
```

For example:

```python
{
    "train": [((words, visual, acoustic), label, sample_id), ...],
    "dev":   [((words, visual, acoustic), label, sample_id), ...],
    "test":  [((words, visual, acoustic), label, sample_id), ...],
}
```

`words` must be a list of word tokens, and acoustic/visual feature arrays must be aligned to those tokens. A `valid` split is also accepted in place of `dev`. Sentiment targets lie in `[-3, 3]`.

Check that the files were downloaded correctly:

```bash
python scripts/check_dataset.py datasets/mosi.pkl
python scripts/check_dataset.py datasets/mosei.pkl
```

Dataset `.pkl` files are not distributed in this repository. Only load pickle files from sources you trust.

## 3. Download pretrained BERT

The language encoder uses **bert-base-uncased**. Official model page:

https://huggingface.co/google-bert/bert-base-uncased

To download the tokenizer, configuration, and pretrained weights into `./bert-base-uncased/`, run the following from the repository root:

```bash
python -c "from transformers import BertTokenizer, BertModel; repo='google-bert/bert-base-uncased'; BertTokenizer.from_pretrained(repo).save_pretrained('bert-base-uncased'); BertModel.from_pretrained(repo).save_pretrained('bert-base-uncased')"
```

After downloading, the directory should contain files such as:

```text
bert-base-uncased/
├── config.json
├── pytorch_model.bin
├── vocab.txt
└── tokenizer_config.json
```

An internet connection is required for the initial download. This model directory is excluded from Git by `.gitignore` and is not included in the source archive.

## 4. Training

Run the following commands from the repository root, after preparing the datasets and BERT files.

### CMU-MOSI

```bash
python main.py --dataset mosi --gpu 0 --seed 0 --n_epochs 50 --bert_path ./bert-base-uncased
```

### CMU-MOSEI

```bash
python main.py --dataset mosei --gpu 0 --seed 0 --n_epochs 50 --train_batch_size 128 --bert_path ./bert-base-uncased
```

`--gpu 0` selects the first visible GPU. For CPU execution, use `--gpu -1` (substantially slower). The default learning rates are `1e-5` for BERT and `2e-5` for the other model parameters.

Best checkpoints are selected by **minimum validation MSE**, not by test accuracy or test F1. Standard training evaluates the held-out test set after training is complete.

## 5. Evaluate a saved checkpoint

```bash
python main.py --dataset mosi --gpu 0 --seed 0 --eval_only
python main.py --dataset mosei --gpu 0 --seed 0 --eval_only
```

The matching checkpoint must be present under `checkpoints/`. To load another checkpoint explicitly, add `--checkpoint /path/to/checkpoint.pth`.

Reported metrics include Acc-2, support-weighted F1, Acc-7, MAE, Pearson correlation (Corr), and MSE. By default, Acc-2 and F1 exclude examples whose ground-truth sentiment is exactly zero; `--include_zero` changes this convention.
