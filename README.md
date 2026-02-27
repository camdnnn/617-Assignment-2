# ENSF 617 — Assignment 2 (Garbage Classification)
Group: 3
Manuja Senanayake , Jack Shenfield, Edmund Yu, Cameron Dunn

This repo contains a multimodal (image + text) garbage classification pipeline, plus SLURM scripts to run training/evaluation on TALC, and saved results/plots.

---

## Quick navigation

- [Setup](#setup)
- [How to run on TALC](#how-to-run-on-talc)
- [Repo structure](#repo-structure)
- [Outputs](#outputs)

---

## Setup

### Option A — Conda (recommended)
- [`environment.yml`](./environment.yml): Conda environment specification (Python + dependencies).
- [`enviroment-setup-guide.md`](./enviroment-setup-guide.md): Step-by-step environment setup instructions.

### Option B — Pip
- [`requirements.txt`](./requirements.txt): Pip dependencies for running the project without Conda.

---

## How to run on TALC

SLURM job scripts live in the repo root:

- [`run_train.slurm`](./run_train.slurm): Submits a training job (runs the training script and writes checkpoints/metrics).
- [`run_smoke.slurm`](./run_smoke.slurm): Submits a small “smoke test” job to verify the pipeline end-to-end quickly.
- [`run_eval_notebook.slurm`](./run_eval_notebook.slurm): Runs the evaluation notebook non-interactively on the cluster.

---

## Repo structure

### Documentation
- [`Garbage-classification-programming.pdf`](./Garbage-classification-programming.pdf): Assignment handout / project description PDF.

### Prototyping
- [`prototyping/dataingestion_labelmapping.ipynb`](./prototyping/dataingestion_labelmapping.ipynb): Notebook used during early prototyping to validate data ingestion and label → index mapping.

### Source code (`src/`)
- [`src/data_ingestion/__init__.py`](./src/data_ingestion/__init__.py): Package init for ingestion utilities.
- [`src/data_ingestion/multimodal_ingestion.py`](./src/data_ingestion/multimodal_ingestion.py): Multimodal dataset ingestion logic (loads samples, applies transforms, prepares text/image inputs, handles label mapping, etc.).

- [`src/models/__init__.py`](./src/models/__init__.py): Package init for model components.
- [`src/models/encoders.py`](./src/models/encoders.py): Encoder modules (e.g., image encoder / text encoder wrappers, feature extraction utilities).
- [`src/models/classifiers.py`](./src/models/classifiers.py): Classification heads / fusion model definition (combines modalities and outputs class logits).

- [`src/training/train.py`](./src/training/train.py): Training entry point (training loop, checkpoint saving, metrics logging, and saving learning-curve plots).

- [`src/evaluation/evaluate.ipynb`](./src/evaluation/evaluate.ipynb): Evaluation notebook (loads a saved checkpoint, runs inference on eval/val set, computes metrics/plots such as confusion matrices).

### Results (`results/`)
- [`results/metrics.csv`](./results/metrics.csv): Logged training/validation metrics per epoch (loss/accuracy, etc.).
- [`results/loss_curve.png`](./results/loss_curve.png): Loss curve plot saved from training.
- [`results/accuracy_curve.png`](./results/accuracy_curve.png): Accuracy curve plot saved from training.

---

## Outputs

Typical outputs produced by training/evaluation include:
- Model checkpoints (e.g., `best.pt`) written by the training run (location depends on `--save-dir` / SLURM script configuration).
- CSV metrics logs for reproducibility ([`results/metrics.csv`](./results/metrics.csv)).
- Training curves ([`results/loss_curve.png`](./results/loss_curve.png), [`results/accuracy_curve.png`](./results/accuracy_curve.png)).
- Evaluation results in [`src/evaluation/evaluate.ipynb`](./src/evaluation/evaluate.ipynb) (e.g., confusion matrix figures, etc).