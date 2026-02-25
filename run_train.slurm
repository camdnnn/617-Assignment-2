cat > ~/ass2/run_train.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name=ensf617_ass2
#SBATCH --output=/home/mahasamilage.senanay/ass2/logs/train_%j.out
#SBATCH --error=/home/mahasamilage.senanay/ass2/logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# ---- Modules ----
module purge
module load conda/base
module load cuda/12.1.1

# ---- Conda env ----
conda activate ensf617-ass2

# ---- Make cluster-safe plotting ----
export MPLBACKEND=Agg

# ---- Optional: keep HF downloads in home cache ----
export HF_HOME=/home/mahasamilage.senanay/.cache/huggingface
export TRANSFORMERS_CACHE=/home/mahasamilage.senanay/.cache/huggingface/transformers

# ---- Run from project root ----
cd /home/mahasamilage.senanay/ass2

# ---- Run training (and evaluation pack) ----
python src/training/train.py \
  --train-path /work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Train \
  --val-path   /work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Val \
  --test-path  /work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Test \
  --save-dir checkpoints \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 2 \
  --eval
EOF