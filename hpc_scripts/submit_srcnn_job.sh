#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --time=01:00:00
#SBATCH --job-name=srcnn_training
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --output=srcnn_training_%j.out
#SBATCH --error=srcnn_training_%j.err

# Load the Python module
module load python/3.8.1

# Activate the virtual environment
source /home/wang.pein/satellite_imagery_super_resolution/hpc_scripts/.venv/bin/activate

# Navigate to the directory containing your Python script
cd /home/wang.pein/satellite_imagery_super_resolution/hpc_scripts

# Run the Python script
python srcnn.py

