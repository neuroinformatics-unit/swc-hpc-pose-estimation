#!/bin/bash 

#SBATCH -J slp_infer # job name
#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 12G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 0-02:00 # time (D-HH:MM)
#SBATCH --gres gpu:1 # request 1 GPU (of any kind)
#SBATCH -o slurm.%N.%j.out # write STDOUT
#SBATCH -e slurm.%N.%j.err # write STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@domain.com 

# Load the SLEAP module
module load SLEAP

# Define directories for SLEAP project and exported training job
SLP_DIR=/ceph/scratch/neuroinformatics-dropoff/SLEAP_HPC_test_data
VIDEO_DIR=$SLP_DIR/videos
SLP_JOB_NAME=labels.v001.slp.training_job_2
SLP_JOB_DIR=$SLP_DIR/$SLP_JOB_NAME

# Go to the job directory
cd $SLP_JOB_DIR
# Make a directory to store the predictions
mkdir -p predictions

# Run the inference command
sleap-track $VIDEO_DIR/M708149_EPM_20200317_165049331-converted.mp4 \
    -m $SLP_JOB_DIR/models/231010_164307.centroid/training_config.json \
    -m $SLP_JOB_DIR/models/231010_164307.centered_instance/training_config.json \
    --gpu auto \
    --tracking.tracker simple \
    --tracking.post_connect_single_breaks 1 \
    -o predictions/labels.v001.slp.predictions.slp \
    --verbosity json \
    --no-empty-frames
