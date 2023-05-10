#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 12G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 0-02:00 # time (D-HH:MM)
#SBATCH --gres gpu:1 # request 1 GPU (of any kind)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ucqfnsi@ucl.ac.uk

# Load the SLEAP module
module load SLEAP

# Define data directory
DATA_DIR=/ceph/scratch/neuroinformatics-dropoff/SLEAP_HPC_test_data
# Define exported job directory
JOB_DIR=$DATA_DIR/labels.v001.slp.training_job

# Go to the job directory
cd $JOB_DIR

# Run the inference command
sleap-track $DATA_DIR/videos/M708149_EPM_20200317_165049331-converted.mp4 \
    -m $JOB_DIR/models/230509_141357.centroid/training_config.json \
    -m $JOB_DIR/models/230509_141357.centered_instance/training_config.json \
    --gpu auto \
    --tracking.tracker none \
    -o labels.v001.slp.predictions.slp \
    --verbosity json \
    --no-empty-frames
