#!/bin/bash
#SBATCH --job-name=kuramoto_end_forever
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=60:00:00
#SBATCH --array=0-10
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eliecer.diazdiaz@aalto.fi

# Load modules in correct order
module load triton/2024.1-gcc
module load cuda/12.2.1

# Activate venv
source $HOME/thesis/.venv/bin/activate

# Print environment info
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "========================================"
nvidia-smi
echo "========================================"
python --version
echo "========================================"

# Move to thesis directory where scripts and data live
cd $HOME/thesis/linespace-original_triton

# One network per array task (11 networks: indices 0-10)
I_START=$SLURM_ARRAY_TASK_ID
I_END=$((SLURM_ARRAY_TASK_ID + 1))

echo "Running networks [$I_START, $I_END)"
echo "========================================"

# Run simulation
python kuramoto_end_forever.py --i-start $I_START --i-end $I_END

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"
