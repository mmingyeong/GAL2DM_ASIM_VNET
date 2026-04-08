#!/bin/bash
#SBATCH -J vnet_tp_lite_plus
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VNET/logs/%x.%A_%a.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VNET/logs/%x.%A_%a.err
#SBATCH -p a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-48:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# ============================================
# Experiment: VNet TP training (Lite / Plus)
# Date: 2026-04-02 (260402)
#
# Goal:
#   Compare performance changes by trainable parameters
#   using Lite and Plus tiers around the Base model.
#
# Fixed settings:
#   input_case=both
#   keep_two_channels=True
#   batch_size=4
#   grad_accum_steps=1
#   num_workers=4
#   pin_memory=True
#   epochs=200
#   amp=True
#   use_augmentation=True
#   target_field=rho
#   train_val_split=0.8
#   seed=42
#   scheduler_type=cosine_warmup
#   warmup_ratio=0.03
#   min_lr_ratio=1e-2
#   patience=20
#   es_delta=0
#
# Tuned training LR for VNet:
#   max_lr=3e-4
#
# Tier cases:
#   array 0 -> Lite  (VNET_BASE=26)
#   array 1 -> Plus  (VNET_BASE=37)
# ============================================

set -e -o pipefail

# -------------------------------
# Environment
# -------------------------------
module purge
module load cuda/12.1.1

source ~/.bashrc
conda activate torch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_MODULE_LOADING=LAZY
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -n 65535

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_VNET"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

# -------------------------------
# Fixed training config
# -------------------------------
MAX_LR=3e-4
MIN_LR_RATIO=1e-2
WARMUP_RATIO=0.03
BATCH_SIZE=4
EPOCHS=200
TRAIN_VAL_SPLIT=0.8
GRAD_ACCUM_STEPS=1
PATIENCE=20
ES_DELTA=0
SEED=42
NUM_WORKERS=4

# -------------------------------
# Tier config (Lite / Plus)
# -------------------------------
TIER_NAMES=(
  "Lite"
  "Plus"
)

EXP_IDS=(
  "260402_tp_lite"
  "260402_tp_plus"
)

VNET_BASE_LIST=(
  "26"
  "37"
)

IDX=${SLURM_ARRAY_TASK_ID}

TIER_NAME=${TIER_NAMES[$IDX]}
EXP_ID=${EXP_IDS[$IDX]}
VNET_BASE=${VNET_BASE_LIST[$IDX]}

RUN_ID="${EXP_ID}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

CKPT_DIR="${PROJECT_ROOT}/results/vnet/tp_training/${EXP_ID}"
LOG_RUN_DIR="${PROJECT_ROOT}/logs/vnet/tp_training/${EXP_ID}"

cd "$PROJECT_ROOT" || { echo "[FATAL] cd failed"; exit 2; }

echo "=== [JOB STARTED] $(date) ==="
echo "EXP_ID            : ${EXP_ID}"
echo "RUN_ID            : ${RUN_ID}"
echo "TIER_NAME         : ${TIER_NAME}"
echo "VNET_BASE         : ${VNET_BASE}"
echo "MAX_LR            : ${MAX_LR}"
echo "MIN_LR_RATIO      : ${MIN_LR_RATIO}"
echo "WARMUP_RATIO      : ${WARMUP_RATIO}"
echo "GRAD_ACCUM_STEPS  : ${GRAD_ACCUM_STEPS}"
echo "BATCH_SIZE        : ${BATCH_SIZE}"
echo "EPOCHS            : ${EPOCHS}"
echo "PATIENCE          : ${PATIENCE}"
echo "USE_AUGMENTATION  : True"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# -------------------------------
# Prepare dirs
# -------------------------------
mkdir -p "${CKPT_DIR}" "${LOG_RUN_DIR}"
LOG_FILE="${LOG_RUN_DIR}/train.log"

# Save config snapshot
cat > "${LOG_RUN_DIR}/run_config.txt" <<EOF
EXP_ID=${EXP_ID}
RUN_ID=${RUN_ID}
TIER_NAME=${TIER_NAME}
VNET_BASE=${VNET_BASE}
MAX_LR=${MAX_LR}
MIN_LR_RATIO=${MIN_LR_RATIO}
WARMUP_RATIO=${WARMUP_RATIO}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
BATCH_SIZE=${BATCH_SIZE}
EPOCHS=${EPOCHS}
TRAIN_VAL_SPLIT=${TRAIN_VAL_SPLIT}
PATIENCE=${PATIENCE}
ES_DELTA=${ES_DELTA}
SEED=${SEED}
NUM_WORKERS=${NUM_WORKERS}
INPUT_CASE=both
KEEP_TWO_CHANNELS=True
AMP=True
USE_AUGMENTATION=True
TARGET_FIELD=rho
PIN_MEMORY=True
SCHEDULER_TYPE=cosine_warmup
EOF

# -------------------------------
# Run training
# -------------------------------
srun python -u -m src.train \
  --yaml_path "${YAML_PATH}" \
  --target_field rho \
  --train_val_split ${TRAIN_VAL_SPLIT} \
  --sample_fraction 1.0 \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --pin_memory True \
  --epochs ${EPOCHS} \
  --scheduler_type cosine_warmup \
  --max_lr ${MAX_LR} \
  --warmup_ratio ${WARMUP_RATIO} \
  --min_lr_ratio ${MIN_LR_RATIO} \
  --patience ${PATIENCE} \
  --es_delta ${ES_DELTA} \
  --grad_accum_steps ${GRAD_ACCUM_STEPS} \
  --input_case both \
  --keep_two_channels \
  --use_augmentation \
  --vnet_base ${VNET_BASE} \
  --ckpt_dir "${CKPT_DIR}" \
  --seed ${SEED} \
  --device cuda \
  --amp \
  --validate_keys False \
  --exclude_list "${PROJECT_ROOT}/filelists/exclude_bad_all.txt" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== [JOB FINISHED] $(date) ==="