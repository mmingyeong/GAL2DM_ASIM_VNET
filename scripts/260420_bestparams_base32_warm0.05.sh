#!/bin/bash
#SBATCH -J vnet_b32_best
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
# Experiment: VNet final best params (Base=32)
# Script name : 260420_bestparams_base32.sh
# Date        : 2026-04-20
#
# Final settings
# - model: VNet (UNet3D)
# - vnet_base=32
# - dtype=float32 + AMP
# - batch_size=4
# - grad_accum_steps=1
# - effective_batch=4
# - epochs=200
# - scheduler=cosine_warmup
# - max_lr=3e-4
# - warmup_ratio=0.05
# - min_lr_ratio=1e-2
# - patience=20
# - es_delta=0
# - input_case=both
# - keep_two_channels=True
# - target_field=rho
# - train_val_split=0.8
# - sample_fraction=1.0
# - use_augmentation=False  ✅ (최종판)
# - validate_keys=False
# - seed=42
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
EXP_ID="260420_bestparams_base32"
RUN_ID="${EXP_ID}_${SLURM_JOB_ID}"

CKPT_DIR="${PROJECT_ROOT}/results/vnet/bestparams_base32/warm0.05/${EXP_ID}"
LOG_RUN_DIR="${PROJECT_ROOT}/logs/vnet/bestparams_base32/warm0.05/${EXP_ID}"

# -------------------------------
# Fixed training config
# -------------------------------
VNET_BASE=32
DTYPE="float32"
TARGET_FIELD="rho"

MAX_LR=3e-4
WARMUP_RATIO=0.05
MIN_LR_RATIO=1e-2

BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
EPOCHS=200
TRAIN_VAL_SPLIT=0.8
SAMPLE_FRACTION=1.0

PATIENCE=20
ES_DELTA=0
SEED=42

NUM_WORKERS=4
PIN_MEMORY=True

INPUT_CASE="both"
KEEP_TWO_CHANNELS="--keep_two_channels"
USE_AUGMENTATION=False
VALIDATE_KEYS=False

EXCLUDE_LIST="${PROJECT_ROOT}/filelists/exclude_bad_all.txt"

cd "$PROJECT_ROOT" || { echo "[FATAL] cd failed"; exit 2; }

echo "=== [JOB STARTED] $(date) ==="
echo "EXP_ID            : ${EXP_ID}"
echo "RUN_ID            : ${RUN_ID}"
echo "VNET_BASE         : ${VNET_BASE}"
echo "DTYPE             : ${DTYPE}"
echo "TARGET_FIELD      : ${TARGET_FIELD}"
echo "MAX_LR            : ${MAX_LR}"
echo "BATCH_SIZE        : ${BATCH_SIZE}"
echo "EPOCHS            : ${EPOCHS}"
echo "PATIENCE          : ${PATIENCE}"
echo "USE_AUGMENTATION  : ${USE_AUGMENTATION}"
echo "DEVICE            : cuda"
echo "AMP               : ON"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(2, 2, device="cuda", dtype=torch.float32)
    print("float32 test dtype:", x.dtype)
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
VNET_BASE=${VNET_BASE}
DTYPE=${DTYPE}
TARGET_FIELD=${TARGET_FIELD}
MAX_LR=${MAX_LR}
BATCH_SIZE=${BATCH_SIZE}
EPOCHS=${EPOCHS}
TRAIN_VAL_SPLIT=${TRAIN_VAL_SPLIT}
SAMPLE_FRACTION=${SAMPLE_FRACTION}
PATIENCE=${PATIENCE}
ES_DELTA=${ES_DELTA}
SEED=${SEED}
NUM_WORKERS=${NUM_WORKERS}
PIN_MEMORY=${PIN_MEMORY}
INPUT_CASE=${INPUT_CASE}
USE_AUGMENTATION=${USE_AUGMENTATION}
VALIDATE_KEYS=${VALIDATE_KEYS}
AMP=ON
EOF

# -------------------------------
# Run training
# -------------------------------
srun python -u -m src.train \
  --yaml_path "${YAML_PATH}" \
  --target_field ${TARGET_FIELD} \
  --train_val_split ${TRAIN_VAL_SPLIT} \
  --sample_fraction ${SAMPLE_FRACTION} \
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
  --ckpt_dir "${CKPT_DIR}" \
  --seed ${SEED} \
  --device cuda \
  --dtype ${DTYPE} \
  --amp \
  --keep_two_channels \
  --validate_keys ${VALIDATE_KEYS} \
  --exclude_list "${EXCLUDE_LIST}" \
  --vnet_base ${VNET_BASE} \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== [JOB FINISHED] $(date) ==="