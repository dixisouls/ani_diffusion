#!/bin/bash
#SBATCH --job-name=ani_diff_test_module_1
#SBATCH --output=/Users/924317352/diffusion/logs/test_module_1_%j.log
#SBATCH --partition=cpucluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00

ENV_NAME="ani_diff"
PROJECT_ROOT="$HOME/diffusion"

echo "================================================"
echo "Slurm Job ID:     $SLURM_JOB_ID"
echo "Running on node:  $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "================================================"

# --- Activate Conda Environment ---
source ~/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -q "^${ENV_NAME} "; then
    conda activate ${ENV_NAME}
    echo "Conda environment activated: $ENV_NAME"
else
    echo "ERROR: Conda environment not found: $ENV_NAME"
    echo "Available conda environments:"
    conda env list
    exit 1
fi

if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "ERROR: Conda activation failed."
    exit 1
fi

echo "Python : $(which python)"
echo "Version: $(python --version)"

# --- Load environment variables ---
cd $PROJECT_ROOT

if [ ! -f "config.env" ]; then
    echo "ERROR: config.env not found in $PROJECT_ROOT"
    exit 1
fi

source config.env
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# --- Run Test ---
echo "Running test_module_1.py..."
python -u tests/test_module_1.py

EXIT_CODE=$?

echo "================================================"
echo "Finished at: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "================================================"

conda deactivate
exit $EXIT_CODE