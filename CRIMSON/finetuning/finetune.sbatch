#!/bin/bash

#SBATCH --job-name=crimson_ft
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH -p gpu_dia
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --mem=240G
#SBATCH -t 30:00:00
#SBATCH -o logs/finetune_%j.out
#SBATCH -e logs/finetune_%j.err

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
echo "Node: $(hostname)"

nvidia-smi

source /home/mob999/miniconda3/etc/profile.d/conda.sh

conda activate cl
echo "Conda env activated: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
which accelerate

# ---- paths ----
SCRIPT_DIR="/home/mob999/CRIMSON/finetuning"
PROJECT_DIR="/home/mob999/rajpurkarlab/home/mob999/models"

TRAIN_JSONL="/home/mob999/rajpurkarlab/datasets/rexgradient/train_data.jsonl"
OUTPUT_DIR="${PROJECT_DIR}/data/finetuned_medgemma/checkpoints"
CACHE_DIR="${PROJECT_DIR}/models/medgemma"

mkdir -p "${OUTPUT_DIR}"

cd "$SCRIPT_DIR"

# ---- launch with accelerate (4 GPUs, bf16) ----
accelerate launch \
    --config_file accelerate_config.yaml \
    finetune_medgemma.py \
        --train_jsonl "$TRAIN_JSONL" \
        --output_dir "$OUTPUT_DIR" \
        --model_id google/medgemma-4b-it \
        --cache_dir "$CACHE_DIR" \
        --max_length 4048 \
        --num_epochs 10 \
        --batch_size 4 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-4 \
        --warmup_ratio 0.05 \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --save_strategy epoch \
        --save_total_limit 3 \
        --dataloader_num_workers 4 \
        --logging_steps 5 \
        --seed 42 \
        --weight_decay 0.05 \
        --num_samples 75000

echo "=== Job $SLURM_JOB_ID completed at $(date) ==="
