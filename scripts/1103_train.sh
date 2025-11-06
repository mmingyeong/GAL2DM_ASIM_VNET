#!/bin/bash
#SBATCH -J vnet_gpu_ps8_e50_both
#SBATCH -o /home/mingyeong/GAL2DM_ASIM_VNET/logs/icase-both/%x.%j.out
#SBATCH -e /home/mingyeong/GAL2DM_ASIM_VNET/logs/icase-both/%x.%j.err
#SBATCH -p h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmingyeong@kasi.re.kr

set -e -o pipefail
module purge
module load cuda/12.1.1
source ~/.bashrc
conda activate torch

CASE_TAG="icase-both"
PROJECT_ROOT="/home/mingyeong/GAL2DM_ASIM_VNET"
YAML_PATH="${PROJECT_ROOT}/etc/asim_paths.yaml"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

CKPT_DIR="${PROJECT_ROOT}/results/vnet/${CASE_TAG}/${RUN_ID}"
LOG_DIR="${PROJECT_ROOT}/logs/${CASE_TAG}/${RUN_ID}"
mkdir -p "${PROJECT_ROOT}/logs/${CASE_TAG}" "${PROJECT_ROOT}/results/vnet/${CASE_TAG}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_MODULE_LOADING=LAZY
export HDF5_USE_FILE_LOCKING=FALSE
ulimit -n 65535

cd "$PROJECT_ROOT" || { echo "[FATAL] cd failed"; exit 2; }

echo "=== [JOB STARTED] $(date) on $(hostname) ==="
which python
python - <<'PY'
import torch, os
print("Torch:", torch.__version__)
print("CUDA:", getattr(torch.version, "cuda", None))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Count:", torch.cuda.device_count())
    print("GPU Name[0]:", torch.cuda.get_device_name(0))
PY
nvidia-smi || echo "nvidia-smi not available"

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.stderr.write("[FATAL] CUDA not available.\n")
    sys.exit(2)
PY

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/vnet_gpu_ps8_e50_both_${RUN_ID}.console.log"
touch "${LOG_FILE}"

echo "Launching training..."
srun python -u -m src.train \
  --yaml_path "${YAML_PATH}" \
  --target_field rho \
  --train_val_split 0.8 \
  --sample_fraction 1.0 \
  --batch_size 4 \
  --num_workers 4 \
  --pin_memory False \
  --epochs 50 \
  --min_lr 1e-4 \
  --max_lr 1e-3 \
  --cycle_length 8 \
  --ckpt_dir "${CKPT_DIR}" \
  --seed 42 \
  --device cuda \
  --input_case both --keep_two_channels \
  --amp \
  --validate_keys False \
  --exclude_list "${PROJECT_ROOT}/filelists/exclude_bad_all.txt" \
  2>&1 | tee -a "${LOG_FILE}"

echo "=== [JOB FINISHED] $(date) ==="
