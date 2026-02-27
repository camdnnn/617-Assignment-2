#!/bin/bash
#SBATCH --job-name=ass2_smoke
#SBATCH --partition=gpu
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:t4:1

module purge
module load conda/base
module load cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ensf617-ass2

export MPLBACKEND=Agg
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Always run from submission directory (project root)
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p checkpoints

echo "=== CUDA CHECK ==="
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "=== START TRAIN ==="
python src/training/train.py \
  --train-path /work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Train \
  --val-path   /work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Val \
  --test-path  /work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Test \
  --save-dir checkpoints \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --eval

echo "=== DONE ==="