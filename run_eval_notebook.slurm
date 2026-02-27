#!/bin/bash
#SBATCH --job-name=ensf617_ass2_eval_nb
#SBATCH --partition=gpu
#SBATCH --output=logs/eval_nb_%j.out
#SBATCH --error=logs/eval_nb_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:t4:1

module purge
module load conda/base
module load cuda/12.1.1

# Make sure conda activate works in batch shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ensf617-ass2

# Headless plotting + consistent HF caches (match training)
export MPLBACKEND=Agg
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Run from the directory where you submitted the job (project root)
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

# Ensure "from models..." works (project-relative)
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH}"

# Execute notebook and save executed copy (with outputs)
python -m nbconvert \
  --to notebook \
  --execute src/evaluation/evaluate.ipynb \
  --output evaluate_executed.ipynb \
  --output-dir src/evaluation \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=ensf617-ass2