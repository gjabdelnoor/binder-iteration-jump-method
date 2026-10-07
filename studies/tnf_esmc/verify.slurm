#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3060:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=tnf-esmc-verify
#SBATCH --output=logs/verify-%j.log
set -euo pipefail
cd "${STUDY_ROOT:?}"
export HF_HOME="$HOME/.cache/huggingface-tnf-study"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PATH="$PWD/.venv/bin:$PATH" JUPYTER_PATH="$PWD/.venv/share/jupyter"
python -u src/verify_run.py --root "$PWD"
python -u src/build_notebook.py --root "$PWD"
printf 'CHECKPOINTS_AND_NOTEBOOK_VERIFIED\n'
