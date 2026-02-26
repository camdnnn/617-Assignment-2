#!/bin/bash
#SBATCH --job-name=ensf617_ass2
#SBATCH --partition=gpu
#SBATCH --output=/home/mahasamilage.senanay/ass2/logs/train_%j.out
#SBATCH --error=/home/mahasamilage.senanay/ass2/logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:t4:1

module purge
module load conda/base
module load cuda/12.1.1

# Make sure conda activate works in batch shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ensf617-ass2

export MPLBACKEND=Agg
export HF_HOME=/home/mahasamilage.senanay/.cache/huggingface
export TRANSFORMERS_CACHE=/home/mahasamilage.senanay/.cache/huggingface/transformers

cd /home/mahasamilage.senanay/ass2

python src/training/train.py \
  --train-path /work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Train \
  --val-path   /work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Val \
  --test-path  /work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Test \
  --save-dir checkpoints \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 2 \
  --eval