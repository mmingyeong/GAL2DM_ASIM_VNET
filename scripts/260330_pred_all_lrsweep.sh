#!/bin/bash
#SBATCH -J vnet_predict_multi
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VNET/logs/vnet_predict_multi.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VNET/logs/vnet_predict_multi.%j.err
#SBATCH -p a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-09:00:00


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
OUT_DIR_BASE="/home/mingyeong/GAL2DM_pred/vnet"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/predict/vnet/${RUN_TS}"
mkdir -p "${LOG_DIR}" "${OUT_DIR_BASE}"

cd "${PROJECT_ROOT}" || { echo "[FATAL] cd failed"; exit 2; }
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =========================================
# Target training runs
# =========================================
CKPT_DIRS=(
  "/home/mingyeong/GAL2DM_ASIM_VNET/results/vnet/lr_1e-3/134176_1"
  "/home/mingyeong/GAL2DM_ASIM_VNET/results/vnet/lr_3e-3/134174_2"
  "/home/mingyeong/GAL2DM_ASIM_VNET/results/vnet/lr_3e-4/134175_0"
)

# =========================================
# Inference configuration
# =========================================
BATCH_SIZE=1
SAMPLE_FRACTION=1.0
INPUT_CASE="both"

AMP_FLAG="--amp"
KEEP_TWO="--keep_two_channels"

# =========================================
# System information
# =========================================
echo "=== [PREDICT START] $(date) on $(hostname) ==="
which python
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi || echo "nvidia-smi not available"

# =========================================
# Helpers
# =========================================
find_checkpoint() {
  local ckpt_dir="$1"
  local model_path=""

  if [ ! -d "${ckpt_dir}" ]; then
    echo ""
    return 0
  fi

  model_path="$(ls -t "${ckpt_dir}"/*best*.pt 2>/dev/null | head -n 1 || true)"
  if [ -z "${model_path}" ]; then
    model_path="$(ls -t "${ckpt_dir}"/*.pt 2>/dev/null | head -n 1 || true)"
  fi

  echo "${model_path}"
}

already_predicted() {
  local out_dir="$1"

  [ -d "${out_dir}" ] || return 1

  # prediction 산출물이 하나라도 있으면 skip
  find "${out_dir}" -maxdepth 1 -type f \( -name "*.h5" -o -name "*.hdf5" -o -name "*.npz" \) | grep -q .
}

# =========================================
# Run prediction for each checkpoint dir
# =========================================
TOTAL=0
DONE=0
SKIPPED=0
FAILED=0

for CKPT_DIR in "${CKPT_DIRS[@]}"; do
  TOTAL=$((TOTAL + 1))

  echo
  echo "============================================================"
  echo "[INFO] Processing CKPT_DIR: ${CKPT_DIR}"
  echo "============================================================"

  if [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR not found: ${CKPT_DIR}"
    FAILED=$((FAILED + 1))
    continue
  fi

  MODEL_PATH="$(find_checkpoint "${CKPT_DIR}")"
  if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] No checkpoint (.pt) found in ${CKPT_DIR}"
    FAILED=$((FAILED + 1))
    continue
  fi

  RUN_STEM="$(basename "${CKPT_DIR}")"
  LR_STEM="$(basename "$(dirname "${CKPT_DIR}")")"
  if [[ "${LR_STEM}" != lr_* ]]; then
    LR_STEM="lr_unknown"
  fi

  PRED_OUT_DIR="${OUT_DIR_BASE}/${LR_STEM}/${RUN_STEM}"
  mkdir -p "${PRED_OUT_DIR}"

  echo "[INFO] Using checkpoint      : ${MODEL_PATH}"
  echo "[INFO] Prediction output dir: ${PRED_OUT_DIR}"

  if already_predicted "${PRED_OUT_DIR}"; then
    echo "[SKIP] Prediction outputs already exist in ${PRED_OUT_DIR}"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  RUN_LOG="${LOG_DIR}/vnet_predict_${LR_STEM}_${RUN_STEM}_job${SLURM_JOB_ID}.log"

  echo "[RUN ] Start prediction for ${LR_STEM}/${RUN_STEM}"
  srun python -u "${PROJECT_ROOT}/src/predict.py" \
    --yaml_path "${YAML_PATH}" \
    --output_dir "${PRED_OUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --device cuda \
    --batch_size ${BATCH_SIZE} \
    --input_case ${INPUT_CASE} \
    ${KEEP_TWO} \
    --sample_fraction ${SAMPLE_FRACTION} \
    ${AMP_FLAG} \
    2>&1 | tee -a "${RUN_LOG}"

  EXIT_CODE=${PIPESTATUS[0]}

  if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "[ERROR] Prediction failed for ${CKPT_DIR} (exit=${EXIT_CODE})"
    FAILED=$((FAILED + 1))
    continue
  fi

  echo "[DONE] Prediction finished for ${LR_STEM}/${RUN_STEM}"
  DONE=$((DONE + 1))
done

# =========================================
# Summary
# =========================================
echo
echo "=== [PREDICT END] $(date) ==="
echo "[SUMMARY] total=${TOTAL} done=${DONE} skipped=${SKIPPED} failed=${FAILED}"

if [ "${FAILED}" -gt 0 ]; then
  exit 1
fi

exit 0