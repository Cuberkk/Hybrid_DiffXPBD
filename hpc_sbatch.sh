#!/bin/bash
#SBATCH --job-name=DiffXPBD_Warp_Train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gpul40s
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --time=7-00:00:00
#SBATCH --mem=80gb
#SBATCH -o /home/kxz365/Github_repo/3D-DiffXPBD-Warp/hpc_logs/%x%N-%j.out
#SBATCH -e /home/kxz365/Github_repo/3D-DiffXPBD-Warp/hpc_logs/%x%N-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kxz365@case.edu
#SBATCH --signal=B:USR1@120

set -euo pipefail

# -----------------------
# Paths / Env
# -----------------------
PROJ_DIR="/home/kxz365/Github_repo/3D-DiffXPBD-Warp"

PYTHON_BIN="${PYTHON_BIN:-/home/zxc703/python_userbases/26warp/bin/python}"

echo "=============================="
echo "JobID   : ${SLURM_JOB_ID}"
echo "Node    : ${SLURMD_NODENAME:-unknown}"
echo "GPU     :"
nvidia-smi -L || true
echo "PROJ    : ${PROJ_DIR}"
echo "=============================="

# -----------------------
# Train
# -----------------------
cd "${PROJ_DIR}"
"${PYTHON_BIN}" -u "${PROJ_DIR}/train.py" -t 100. -cnt "up" -e 1000 -optidx 3 0 -stpidx 2 -cpts 10 -optst 2 -opteridx 1