#!/bin/bash
#SBATCH --job-name=mirai-ddp-finetune
#SBATCH --output=/groups/dk3360_gp/hs3522/Mirai_DBT/Mirai/logs/%x_%j_%t.out
#SBATCH --error=/groups/dk3360_gp/hs3522/Mirai_DBT/Mirai/logs/%x_%j_%t.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH -p gpu
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#
# Multi-node multi-GPU distributed finetuning for Mirai.
# Each node has exactly 2 GPUs. Total GPUs = nodes * 2.
#
# Usage:
#   sbatch scripts/slurm_finetune.sh
#   sbatch --nodes=8 scripts/slurm_finetune.sh       # Override to 8 nodes (16 GPUs)
#   sbatch scripts/slurm_finetune.sh --init_lr 1e-5   # Pass extra flags to main_ddp.py
#
# The script uses srun to launch one process per GPU.
# SLURM environment variables (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID)
# are automatically set by srun and read by scripts/main_ddp.py.

#set -euo pipefail

# ============================================================
# Networking: set MASTER_ADDR and MASTER_PORT
# ============================================================
# Master node setup
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS

# Network configuration (based on your test)
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
#export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# PyTorch settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print debug info
echo "===== DISTRIBUTED TRAINING INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Total GPUs: $SLURM_NTASKS"
echo "GPUs per node: $SLURM_NTASKS_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "======================================"



# ============================================================
# Create log directory
# ============================================================

# ============================================================
# Default training flags (can override via $@ or SLURM_EXTRA_FLAGS)
# ============================================================
DEFAULT_FLAGS=(
    --model_name mirai_full
    --img_encoder_snapshot  /groups/dk3360_gp/hs3522/Mirai_DBT/Mirai/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p
    --transformer_snapshot  /groups/dk3360_gp/hs3522/Mirai_DBT/Mirai/snapshots/mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p
    --batch_size 32
    --batch_splits 4
    --cuda
    --distributed
    --dataset csv_mammo_risk_all_full_future
    --metadata_path /groups/dk3360_gp/hs3522/Mirai_DBT/Mirai/demo/custom_metadata.csv
    --img_mean 52.7232
    --img_size 1024 1024
    --img_std 60.3772
    --num_workers 4
    --train
    --dev
    --test
    --init_lr 1e-5
    --epochs 15
    --dropout 0.1
    --weight_decay 5e-05
    --num_gpus 1
    --tuning_metric c_index
    --save_dir /groups/dk3360_gp/hs3522/Mirai_DBT/Mirai/snapshot_distributed/
    --results_path /groups/dk3360_gp/hs3522/Mirai_DBT/Mirai/logs/distributed_finetune
    --num_slices 21
    --slice_policy grouped
    --slice_jitter 2
    --slice_encoder_chunk_size 2
    --depth_stats_dropout 0.2
)

# Merge defaults with any extra flags passed on the command line
ALL_FLAGS=("${DEFAULT_FLAGS[@]}" "$@")

# ============================================================
# Launch: one process per GPU via srun
# ============================================================
srun python -u /groups/dk3360_gp/hs3522/Mirai_DBT/Mirai/scripts/main_ddp.py "${ALL_FLAGS[@]}"

echo "============================================="
echo "Distributed finetuning complete."
echo "============================================="
