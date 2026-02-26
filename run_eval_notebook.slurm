#!/bin/bash
#SBATCH --job-name=ensf617_ass2_eval_nb
#SBATCH --partition=gpu
#SBATCH --output=/home/mahasamilage.senanay/ass2/logs/eval_nb_%j.out
#SBATCH --error=/home/mahasamilage.senanay/ass2/logs/eval_nb_%j.err
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
export HF_HOME=/home/mahasamilage.senanay/.cache/huggingface
export TRANSFORMERS_CACHE=/home/mahasamilage.senanay/.cache/huggingface/transformers

cd /home/mahasamilage.senanay/ass2
mkdir -p logs

# Ensure "from models..." works
export PYTHONPATH="/home/mahasamilage.senanay/ass2/src:$PYTHONPATH"

# Execute notebook and save executed copy (with outputs)
python -m nbconvert \
  --to notebook \
  --execute src/evaluation/evaluate.ipynb \
  --output evaluate_executed.ipynb \
  --output-dir src/evaluation \
  --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=ensf617-ass2