#!/bin/bash
#SBATCH -J vnet_predict_fp32_fp64
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VNET/logs/vnet_predict_fp32_fp64.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VNET/logs/vnet_predict_fp32_fp64.%j.err
#SBATCH -p a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-12:00:00

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
OUT_DIR_BASE="/home/mingyeong/GAL2DM_pred/vnet_fp_compare"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/predict/vnet_fp_compare/${RUN_TS}"

mkdir -p "${LOG_DIR}" "${OUT_DIR_BASE}"
cd "${PROJECT_ROOT}" || exit 2

# =========================================
# Target models (🔥 핵심)
# =========================================
CKPT_DIRS=(
  "/home/mingyeong/GAL2DM_ASIM_VNET/results/vnet/warmup_accum/260331_exp1"   # float32
  "/home/mingyeong/GAL2DM_ASIM_VNET/results/vnet/fp64/260413_vnet_fp64"      # float64
)

DTYPES=(
  "float32"
  "float64"
)

# =========================================
# Config
# =========================================
BATCH_SIZE=1
INPUT_CASE="both"
KEEP_TWO="--keep_two_channels"
SAMPLE_FRACTION=1.0

# =========================================
# GPU info
# =========================================
echo "=== [PREDICT START] $(date) ==="
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
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
  echo "$path"
}

# =========================================
# Run
# =========================================
TOTAL=0
DONE=0
FAILED=0

for i in "${!CKPT_DIRS[@]}"; do
  CKPT_DIR="${CKPT_DIRS[$i]}"
  DTYPE="${DTYPES[$i]}"
  TOTAL=$((TOTAL+1))

  echo
  echo "==========================================="
  echo "[INFO] CKPT_DIR : ${CKPT_DIR}"
  echo "[INFO] DTYPE    : ${DTYPE}"
  echo "==========================================="

  MODEL_PATH="$(find_checkpoint "${CKPT_DIR}")"

  if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] No checkpoint found"
    FAILED=$((FAILED+1))
    continue
  fi

  EXP_NAME="$(basename "${CKPT_DIR}")"
  OUT_DIR="${OUT_DIR_BASE}/${DTYPE}/${EXP_NAME}"

  mkdir -p "${OUT_DIR}"

  # AMP 설정
  if [ "${DTYPE}" == "float32" ]; then
    AMP_FLAG="--amp"
  else
    AMP_FLAG=""   # 🔥 float64는 AMP 끔
  fi

  RUN_LOG="${LOG_DIR}/predict_${DTYPE}_${EXP_NAME}.log"

  echo "[RUN ] ${DTYPE} prediction..."

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
    ${AMP_FLAG} \
    2>&1 | tee -a "${RUN_LOG}"

  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[ERROR] Failed (${DTYPE})"
    FAILED=$((FAILED+1))
    continue
  fi

  echo "[DONE] ${DTYPE} prediction done"
  DONE=$((DONE+1))
done

# =========================================
# Summary
# =========================================
echo
echo "=== [PREDICT END] $(date) ==="
echo "TOTAL=${TOTAL} DONE=${DONE} FAILED=${FAILED}"

if [ "${FAILED}" -gt 0 ]; then
  exit 1
fi

exit 0