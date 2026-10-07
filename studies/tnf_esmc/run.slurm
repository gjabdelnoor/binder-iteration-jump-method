#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_3060:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --job-name=tnf-esmc-study
#SBATCH --output=logs/study-%j.log
set -euo pipefail
cd "${STUDY_ROOT:?}"
export HF_HOME="$HOME/.cache/huggingface-tnf-study"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PATH="$PWD/.venv/bin:$PATH"
export JUPYTER_PATH="$PWD/.venv/share/jupyter"
python -m unittest discover -s src -p test_study.py -v 2>&1 | tee results/unit_tests.log
python -u src/study.py --root "$PWD"
python -m ipykernel install --prefix "$PWD/.venv" --name python3 --display-name 'Python 3'
python -u src/build_notebook.py --root "$PWD"
printf 'ALL_REQUESTED_STUDY_STAGES_COMPLETE\n'
