#!/bin/bash
#SBATCH -J vnet_seed_ens
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VNET/logs/%x.%A_%a.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VNET/logs/%x.%A_%a.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 0-48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr
#SBATCH --array=0-9

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
# Random seed configs
# -------------------------------
SEED_LIST=(1049 2037 3181 4267 5393 6421 7559 8617 9733 10891)
SEED=${SEED_LIST[$SLURM_ARRAY_TASK_ID]}

# Main benchmark V-Net setting
VNET_BASE=32

# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_VNET"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

# -------------------------------
# Run naming
# -------------------------------
EXP_ID="260611_vnet_seed${SEED}"
RUN_ID="${EXP_ID}_${SLURM_JOB_ID}"

CKPT_DIR="${PROJECT_ROOT}/results/vnet/random_seed_ensemble/${EXP_ID}"
LOG_RUN_DIR="${PROJECT_ROOT}/logs/vnet/random_seed_ensemble/${EXP_ID}"

# -------------------------------
# Fixed training config
# -------------------------------
DTYPE="float32"
TARGET_FIELD="rho"

MAX_LR=3e-4
WARMUP_RATIO=0.03
MIN_LR_RATIO=1e-2

BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
EPOCHS=200

TRAIN_VAL_SPLIT=0.8
SAMPLE_FRACTION=1.0

PATIENCE=20
ES_DELTA=0

NUM_WORKERS=4
PIN_MEMORY=True

INPUT_CASE="both"
USE_AUGMENTATION=False
VALIDATE_KEYS=False

EXCLUDE_LIST="${PROJECT_ROOT}/filelists/exclude_bad_all.txt"

# -------------------------------
# Move to project
# -------------------------------
cd "$PROJECT_ROOT" || {
    echo "[FATAL] cd failed"
    exit 2
}

# -------------------------------
# Logging
# -------------------------------
echo "======================================================"
echo "JOB STARTED : $(date)"
echo "SLURM JOB   : ${SLURM_JOB_ID}"
echo "ARRAY ID    : ${SLURM_ARRAY_TASK_ID}"
echo "EXP_ID      : ${EXP_ID}"
echo "RUN_ID      : ${RUN_ID}"
echo "SEED        : ${SEED}"
echo "VNET_BASE   : ${VNET_BASE}"
echo "======================================================"

python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(2, 2, device="cuda", dtype=torch.float32)
    print("float32 dtype:", x.dtype)
PY

# -------------------------------
# Prepare dirs
# -------------------------------
mkdir -p "${CKPT_DIR}"
mkdir -p "${LOG_RUN_DIR}"

LOG_FILE="${LOG_RUN_DIR}/train.log"

# -------------------------------
# Save config snapshot
# -------------------------------
cat > "${LOG_RUN_DIR}/run_config.txt" <<EOF
EXP_ID=${EXP_ID}
RUN_ID=${RUN_ID}

MODEL=VNET
RANDOM_SEED_ENSEMBLE=True
ENSEMBLE_MEMBER=${SLURM_ARRAY_TASK_ID}
SEED=${SEED}

VNET_BASE=${VNET_BASE}

DTYPE=${DTYPE}
TARGET_FIELD=${TARGET_FIELD}

MAX_LR=${MAX_LR}
WARMUP_RATIO=${WARMUP_RATIO}
MIN_LR_RATIO=${MIN_LR_RATIO}

BATCH_SIZE=${BATCH_SIZE}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}
EPOCHS=${EPOCHS}

TRAIN_VAL_SPLIT=${TRAIN_VAL_SPLIT}
SAMPLE_FRACTION=${SAMPLE_FRACTION}

PATIENCE=${PATIENCE}
ES_DELTA=${ES_DELTA}

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

echo "======================================================"
echo "JOB FINISHED : $(date)"
echo "======================================================"