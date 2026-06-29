#!/bin/bash
#SBATCH -J vnet_fp64
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VNET/logs/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VNET/logs/%x.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 0-48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# ============================================
# Experiment: Vnet float64 training
# Date: 2026-04-13
#
# Fixed:
# - dtype=float64
# - batch_size=4
# - epochs=200
# - scheduler_type=cosine_warmup
# - max_lr=3e-4
# - warmup_ratio=0.03
# - min_lr_ratio=1e-2
# - patience=15
# - input_case=both
# - keep_two_channels=True
# - no augmentation
#
# Notes:
# - float64 training is much slower and uses more memory than float32
# - AMP is intentionally OFF
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
# Run / experiment naming
# -------------------------------
EXP_ID="260413_vnet_fp64"
RUN_ID="${EXP_ID}_${SLURM_JOB_ID}"

CKPT_DIR="${PROJECT_ROOT}/results/vnet/fp64/${EXP_ID}"
LOG_RUN_DIR="${PROJECT_ROOT}/logs/vnet/fp64/${EXP_ID}"

# -------------------------------
# Fixed training config
# -------------------------------
DTYPE="float64"
TARGET_FIELD="rho"

MAX_LR=3e-4
WARMUP_RATIO=0.03
MIN_LR_RATIO=1e-2

BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
EPOCHS=200
TRAIN_VAL_SPLIT=0.8

PATIENCE=15
ES_DELTA=0
SEED=42

NUM_WORKERS=4
PIN_MEMORY=True

INPUT_CASE="both"
KEEP_TWO_CHANNELS="--keep_two_channels"

cd "$PROJECT_ROOT" || { echo "[FATAL] cd failed"; exit 2; }

echo "=== [JOB STARTED] $(date) ==="
echo "EXP_ID            : ${EXP_ID}"
echo "RUN_ID            : ${RUN_ID}"
echo "DTYPE             : ${DTYPE}"
echo "TARGET_FIELD      : ${TARGET_FIELD}"
echo "MAX_LR            : ${MAX_LR}"
echo "WARMUP_RATIO      : ${WARMUP_RATIO}"
echo "MIN_LR_RATIO      : ${MIN_LR_RATIO}"
echo "BATCH_SIZE        : ${BATCH_SIZE}"
echo "GRAD_ACCUM_STEPS  : ${GRAD_ACCUM_STEPS}"
echo "EPOCHS            : ${EPOCHS}"
echo "PATIENCE          : ${PATIENCE}"
echo "DEVICE            : cuda"
echo "AMP               : OFF"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(2, 2, device="cuda", dtype=torch.float64)
    print("float64 test dtype:", x.dtype)
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
DTYPE=${DTYPE}
TARGET_FIELD=${TARGET_FIELD}
MAX_LR=${MAX_LR}
WARMUP_RATIO=${WARMUP_RATIO}
MIN_LR_RATIO=${MIN_LR_RATIO}
BATCH_SIZE=${BATCH_SIZE}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
EPOCHS=${EPOCHS}
TRAIN_VAL_SPLIT=${TRAIN_VAL_SPLIT}
PATIENCE=${PATIENCE}
ES_DELTA=${ES_DELTA}
SEED=${SEED}
NUM_WORKERS=${NUM_WORKERS}
PIN_MEMORY=${PIN_MEMORY}
INPUT_CASE=${INPUT_CASE}
AMP=OFF
EOF

# -------------------------------
# Run training
# -------------------------------
srun python -u -m src.train \
  --yaml_path "${YAML_PATH}" \
  --target_field ${TARGET_FIELD} \
  --train_val_split ${TRAIN_VAL_SPLIT} \
  --sample_fraction 1.0 \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --pin_memory ${PIN_MEMORY} \
  --epochs ${EPOCHS} \
  --scheduler_type cosine_warmup \
  --max_lr ${MAX_LR} \
  --warmup_ratio ${WARMUP_RATIO} \
  --min_lr_ratio ${MIN_LR_RATIO} \
  --patience ${PATIENCE} \
  --es_delta ${ES_DELTA} \
  --grad_accum_steps ${GRAD_ACCUM_STEPS} \
  --input_case ${INPUT_CASE} \
  ${KEEP_TWO_CHANNELS} \
  --ckpt_dir "${CKPT_DIR}" \
  --seed ${SEED} \
  --device cuda \
  --dtype ${DTYPE} \
  --validate_keys False \
  --exclude_list "${PROJECT_ROOT}/filelists/exclude_bad_all.txt" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== [JOB FINISHED] $(date) ==="