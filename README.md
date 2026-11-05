# CVSG: Coalition Value and Synergy-Guided Multimodal Fusion for Sentiment Analysis

CVSG is a supervised multimodal sentiment analysis framework combining
linguistic, acoustic, and visual signals. Its central mechanism learns
modality allocation and pairwise corrections from predictive coalition values.

## Architecture

- **Distributional Modality Encoding**: BERT text features and audio/visual
  Transformer features are represented with modality-specific Gaussian
  bottlenecks. Token-level representations are pooled into modality features,
  alongside posterior-scale descriptors.
- **Value and Synergy Fusion**: learned utility predictors allocate modality
  content, while predicted pair gains control feature interaction proposals.
  Their combination passes through residual fusion and a fused Gaussian
  bottleneck to produce continuous sentiment predictions.
- **Coalition Value Supervision**: during training, a shared auxiliary head
  evaluates seven nonempty modality subsets. Prediction-loss-derived Shapley
  marginal contributions and second-order pair differences supervise the
  utility and pair-gain estimators. The coalition branch is not required
  for standard inference.

## Repository structure

```text
CVSG/
├── main.py                 # Training, validation, evaluation, checkpoints
├── model.py                # CVSG neural network and auxiliary objectives
├── global_configs.py      # Input feature dimensions and device defaults
├── modules/                # Attention and positional encoding components
├── datasets/README.md     # Input data format
├── requirements.txt       # Python dependencies
├── checkpoints/            # Saved checkpoints (excluded from Git)
├── results/                # Experiment outputs (excluded from Git)
└── THIRD_PARTY_NOTICES.md # Source and dependency provenance
```

## Requirements and setup

Python 3.10–3.11 and a compatible PyTorch installation are recommended.
For GPU use, install a PyTorch build matching your CUDA runtime, then run:

```bash
python -m pip install -r requirements.txt
```

Prepare a local `bert-base-uncased/` directory containing configuration,
tokenizer, and pretrained weights, or allow the model files to be retrieved
through your configured Hugging Face environment. BERT weights are not
included in this repository.

Place `mosi.pkl` or `mosei.pkl` under `datasets/` in the format described
in `datasets/README.md`. The datasets are not included in this repository.

## Training

```bash
# CMU-MOSI
python main.py --dataset mosi --gpu 0 --seed 0 --n_epochs 50 --bert_path ./bert-base-uncased

# CMU-MOSEI
python main.py --dataset mosei --gpu 0 --seed 0 --n_epochs 50 --train_batch_size 128 --bert_path ./bert-base-uncased
```

Run additional seeds independently and report mean and standard deviation
across completed runs. The default checkpoint is chosen using minimum
validation MSE; `--select_metric mae` switches checkpoint selection to
minimum validation MAE. The held-out test set is evaluated after training
by default.

## Evaluation

```bash
python main.py --dataset mosi --gpu 0 --seed 0 --eval_only
```

The evaluation process reads the saved checkpoint for the corresponding
model configuration and seed. Metrics include Acc-2, support-weighted F1,
Acc-7, MAE, Corr, and MSE. Acc-2 and F1 exclude samples with exactly zero
ground-truth sentiment unless `--include_zero` is specified.

## Available training variants

The following options are implemented as CLI switches:

```bash
--disable_coalition            # Disable coalition-based training objectives
--disable_value                # Use uniform modality allocation
--disable_synergy              # Disable pair corrections
--disable_uncertainty          # Remove scale descriptors
--disable_synergy_supervision  # Disable direct pair-gain supervision
--kl_weight 0.001
--coalition_weight 0.08
--value_weight 0.04
--synergy_weight 0.02
--utility_temperature 1.5
--pair_strength 0.65
```

These switches do not implement every comparison from the manuscript.
In particular, scale-only routing, a constant pair gate, and uniform
pair strength require separate controlled implementations before they
can be reported as reproducible ablations.

## Experimental integrity

- Choose checkpoints with validation metrics, not test metrics.
- Use exactly the same preprocessing, seeds, and hyperparameters when
  comparing model variants.
- Do not present a logged result as reproduced unless the dataset,
  pretrained weights, and runtime configuration are recorded.
- This package has been checked for Python syntax and source consistency;
  complete benchmark training has not been independently verified here.

## Dataset and code permissions

External datasets, model weights, and third-party components have their
own redistribution terms. See `THIRD_PARTY_NOTICES.md` before publishing.


## Multiple-seed evaluation and data validation

```bash
python scripts/check_dataset.py datasets/mosi.pkl
python scripts/run_five_seeds.py --dataset mosi --gpu 0 --n_epochs 50 --bert_path ./bert-base-uncased
```

These scripts do not supply datasets, pretrained weights, or unimplemented ablation configurations.
