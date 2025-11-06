#!/bin/bash
#SBATCH -J unet3d_predict_ps8
#SBATCH -o logs/unet3d_predict_ps8.%j.out
#SBATCH -e logs/unet3d_predict_ps8.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# =========================================
# Environment
# =========================================
set -e -o pipefail

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

# 훈련 산출물 디렉터리 (수정 가능)
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/results/vnet/icase-both/28845}"

# 출력 디렉터리
OUT_DIR_BASE="${PROJECT_ROOT}/results/unet_predictions"

# 로그 디렉터리
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/predict/${RUN_TS}"
mkdir -p "${LOG_DIR}" "${OUT_DIR_BASE}"

cd "${PROJECT_ROOT}" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# =========================================
# Checkpoint auto-detect
# =========================================
if [ -z "${MODEL_PATH:-}" ]; then
  if [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] CKPT_DIR not found: ${CKPT_DIR}"
    exit 1
  fi
  BEST_CKPT="$(ls -t ${CKPT_DIR}/*best*.pt 2>/dev/null | head -n 1)"
  if [ -n "${BEST_CKPT}" ]; then
    MODEL_PATH="${BEST_CKPT}"
  else
    MODEL_PATH="$(ls -t ${CKPT_DIR}/*.pt 2>/dev/null | head -n 1)"
  fi
  if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] No .pt file in ${CKPT_DIR}"
    exit 1
  fi
fi
echo "[INFO] Using checkpoint: ${MODEL_PATH}"

RUN_STEM="$(basename "${CKPT_DIR}")"
PRED_OUT_DIR="${OUT_DIR_BASE}/${RUN_STEM}"
mkdir -p "${PRED_OUT_DIR}"

# =========================================
# Inference configuration
# =========================================
BATCH_SIZE=1
AMP_FLAG="--amp"
SAMPLE_FRACTION=1.0

INPUT_CASE="both"
KEEP_TWO="--keep_two_channels"

# =========================================
# System information
# =========================================
echo "=== [PREDICT START] $(date) on $(hostname) ==="
which python
python - <<'PY'
import torch, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi || echo "nvidia-smi not available"

# =========================================
# Run prediction
# =========================================
srun python -u "${PROJECT_ROOT}/src/predict.py" \
  --yaml_path "${YAML_PATH}" \
  --output_dir "${PRED_OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --device "cuda" \
  --batch_size ${BATCH_SIZE} \
  --input_case ${INPUT_CASE} \
  ${KEEP_TWO} \
  --sample_fraction ${SAMPLE_FRACTION} \
  ${AMP_FLAG} \
  2>&1 | tee -a "${LOG_DIR}/unet3d_predict_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== [PREDICT END] $(date) (exit=${EXIT_CODE}) ==="
exit ${EXIT_CODE}
