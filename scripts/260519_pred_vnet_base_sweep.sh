#!/bin/bash
#SBATCH -J vnet_base_pred
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VNET/logs/vnet_base_pred_%A_%a.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VNET/logs/vnet_base_pred_%A_%a.err
#SBATCH -p a40
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-12:00:00
#SBATCH --array=0-2

set -e -o pipefail

# =========================================
# Environment
# =========================================
module purge
module load cuda/12.1.1

source ~/.bashrc
conda activate torch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_MODULE_LOADING=LAZY
ulimit -n 65535

# =========================================
# Paths
# =========================================
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_VNET"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

BASE_LIST=(32 64 128)
BASE="${BASE_LIST[$SLURM_ARRAY_TASK_ID]}"

CKPT_DIR="${PROJECT_ROOT}/results/vnet/base_sweep/260519_vnet_base${BASE}"
EXP_NAME="$(basename "${CKPT_DIR}")"

OUT_DIR="/home/mingyeong/GAL2DM_pred/vnet_base_sweep/${EXP_NAME}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/predict/vnet_base_sweep/${RUN_TS}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
cd "${PROJECT_ROOT}" || exit 2

# =========================================
# Config
# =========================================
BATCH_SIZE=1
INPUT_CASE="both"
KEEP_TWO="--keep_two_channels"
SAMPLE_FRACTION=1.0
DTYPE="float32"
AMP_FLAG="--amp"

echo "=== [PREDICT START] $(date) ==="
echo "[INFO] BASE      : ${BASE}"
echo "[INFO] CKPT_DIR  : ${CKPT_DIR}"
echo "[INFO] OUT_DIR   : ${OUT_DIR}"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# =========================================
# Helper
# =========================================
find_checkpoint() {
  local dir="$1"
  local path=""

  path="$(ls -t "${dir}"/*best*.pt 2>/dev/null | head -n 1 || true)"
  if [ -z "$path" ]; then
    path="$(ls -t "${dir}"/*.pt 2>/dev/null | head -n 1 || true)"
  fi
  if [ -z "$path" ]; then
    path="$(ls -t "${dir}"/*best*.pth 2>/dev/null | head -n 1 || true)"
  fi
  if [ -z "$path" ]; then
    path="$(ls -t "${dir}"/*.pth 2>/dev/null | head -n 1 || true)"
  fi

  echo "$path"
}

MODEL_PATH="$(find_checkpoint "${CKPT_DIR}")"

if [ -z "${MODEL_PATH}" ]; then
  echo "[ERROR] No checkpoint found in ${CKPT_DIR}"
  exit 1
fi

echo "[INFO] MODEL_PATH: ${MODEL_PATH}"

RUN_LOG="${LOG_DIR}/predict_${EXP_NAME}.log"

# =========================================
# Run
# =========================================
srun python -u -m src.predict \
  --yaml_path "${YAML_PATH}" \
  --output_dir "${OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device cuda \
  --batch_size ${BATCH_SIZE} \
  --input_case ${INPUT_CASE} \
  ${KEEP_TWO} \
  --sample_fraction ${SAMPLE_FRACTION} \
  --dtype ${DTYPE} \
  --base_channels ${BASE} \
  ${AMP_FLAG} \
  2>&1 | tee -a "${RUN_LOG}"

STATUS=${PIPESTATUS[0]}

echo
echo "=== [PREDICT END] $(date) ==="

if [ "${STATUS}" -ne 0 ]; then
  echo "[ERROR] Prediction failed for BASE=${BASE}"
  exit 1
fi

echo "[DONE] Prediction completed successfully for BASE=${BASE}"
echo "[DONE] Output saved to: ${OUT_DIR}"
exit 0