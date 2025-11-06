#!/bin/bash
#SBATCH -J vnet3d_eval_fast
#SBATCH -o logs/vnet3d_eval_fast.%j.out
#SBATCH -e logs/vnet3d_eval_fast.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

# ========================================
# Usage:
#   sbatch /home/mingyeong/GAL2DM_ASIM_VNET/scripts/1103_eval_fast_vnet.sh \
#     --pred_dir /home/mingyeong/GAL2DM_ASIM_VNET/results/unet_predictions/28845/icase-both-keep2
# ========================================

# -------- Parse args (simplified) --------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 --pred_dir <prediction_directory>"
  exit 1
fi

PRED_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pred_dir) PRED_DIR="$2"; shift 2;;
    *) echo "[WARN] Unknown arg: $1"; shift;;
  esac
done

if [[ -z "${PRED_DIR}" ]]; then
  echo "[ERROR] --pred_dir is required." >&2
  exit 1
fi

# Normalize and validate
PRED_DIR="${PRED_DIR%/}"
if [[ ! -d "${PRED_DIR}" ]]; then
  echo "[ERROR] pred_dir not found: ${PRED_DIR}" >&2
  exit 1
fi

# ========================================
# Environment setup
# ========================================
module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ========================================
# Paths
# ========================================
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_VNET"
EVAL_PY="${PROJECT_ROOT}/src/eval_compare.py"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"

if [[ ! -f "${EVAL_PY}" ]]; then
  echo "[ERROR] evaluator not found: ${EVAL_PY}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}" || exit 1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ========================================
# Alex template (DIRECT, fixed template with {idx})
#   NOTE: adjust zero-padding if needed: {idx:03d} → {idx:05d}, etc.
# ========================================
ALEX_TPL="/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/predictions/test_{idx}_*rho*.npy"
# If some files don't have 'rho' in the name, use this instead:
# ALEX_TPL="/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/predictions/test_{idx}_*.npy"

# ========================================
# Output configuration
# ========================================
CASE_LABEL="$(basename "${PRED_DIR}")"
TS_FROM_PRED="$(basename "$(dirname "${PRED_DIR}")")"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="results/vnet_eval/${RUN_TS}/${CASE_LABEL}"
mkdir -p "logs" "${OUT_ROOT}"

# Auto-discover loss CSV (VNet 전용)
CAND_DIR="${PROJECT_ROOT}/results/vnet/${TS_FROM_PRED}"
LOSS_CSV_RESOLVED=""
if [[ -d "${CAND_DIR}" ]]; then
  LOSS_CSV_RESOLVED="$(ls -1 "${CAND_DIR}"/*log.csv 2>/dev/null | head -n1 || true)"
fi

# Training LOG (학습 시간/에폭/장비 파싱용 · 고정 경로 전달)
TRAIN_LOG="/home/mingyeong/GAL2DM_ASIM_VNET/logs/icase-both/vnet_gpu_ps8_e50_both.28845.out"
if [[ ! -f "${TRAIN_LOG}" ]]; then
  echo "[WARN] train_log not found: ${TRAIN_LOG} (GPU/시간/에폭 파싱은 스킵될 수 있음)"
fi

# ========================================
# Eval parameters (defaults)
# ========================================
SLICE_AXIS=2
SLICE_INDEX=center
MAP_COUNT=5
KS_GLOBAL_CAP=200000
JOINT_SAMPLE=50000
PDF_BINS=120
JOINT_BINS=120
VOXEL_SIZE=$(python - <<'PY'
print(205.0/250.0)
PY
)
RMAX=10.0
N_R_BINS=24

# ========================================
# Print configuration
# ========================================
echo "=== [EVAL FAST START] $(date) on $(hostname) ==="
echo "[INFO] pred_dir       : ${PRED_DIR}"
echo "[INFO] out_root       : ${OUT_ROOT}"
echo "[INFO] yaml_path      : ${YAML_PATH}"
echo "[INFO] alex_tpl       : ${ALEX_TPL}"
echo "[INFO] loss_csv       : ${LOSS_CSV_RESOLVED:-<auto-not-found>}"
echo "[INFO] train_log      : ${TRAIN_LOG}"
echo "[INFO] map_count      : ${MAP_COUNT}"
echo "[INFO] ks_global_cap  : ${KS_GLOBAL_CAP}"
which python
python - <<'PY'
import torch, os
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
nvidia-smi || true

# ========================================
# Run evaluator
# ========================================
srun python -u "${EVAL_PY}" \
  --yaml_path "${YAML_PATH}" \
  --pred_dir "${PRED_DIR}" \
  --alex_tpl "${ALEX_TPL}" \
  --out_dir "${OUT_ROOT}" \
  --slice_axis ${SLICE_AXIS} \
  --slice_index ${SLICE_INDEX} \
  --map_count ${MAP_COUNT} \
  --ks_global_cap ${KS_GLOBAL_CAP} \
  --joint_sample ${JOINT_SAMPLE} \
  --pdf_bins ${PDF_BINS} \
  --joint_bins ${JOINT_BINS} \
  --voxel_size ${VOXEL_SIZE} \
  --rmax ${RMAX} \
  --n_r_bins ${N_R_BINS} \
  --label_pred "${CASE_LABEL}" \
  $( [ -n "${LOSS_CSV_RESOLVED}" ] && echo --loss_csv "${LOSS_CSV_RESOLVED}" ) \
  $( [ -f "${TRAIN_LOG}" ] && echo --train_log "${TRAIN_LOG}" ) \
  --save_latex \
  2>&1 | tee -a "logs/vnet3d_eval_fast_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== [EVAL FAST END] $(date) (exit=${EXIT_CODE}) ==="
exit ${EXIT_CODE}
