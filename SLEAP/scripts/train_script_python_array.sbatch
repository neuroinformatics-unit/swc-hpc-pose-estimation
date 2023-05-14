#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 12G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 0-04:00 # time (D-HH:MM)
#SBATCH --gres gpu:rtx2080:1 # request specific GPU type (rtx2080)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.sirmpilatze@ucl.ac.uk
#SBATCH --array=0-2

# Load the SLEAP module
module load SLEAP

# Define directory for Python scripts
DATA_DIR=/ceph/scratch/neuroinformatics-dropoff/SLEAP_HPC_test_data
CODE_DIR=$DATA_DIR/swc-hpc-pose-estimation/SLEAP/scripts
cd $CODE_DIR

# Run the Python script across batch sizes 2, 4, 8
ARGS=(2 4 8)
python run_sleap_training.py --batch-size "${ARGS[$SLURM_ARRAY_TASK_ID]}"
