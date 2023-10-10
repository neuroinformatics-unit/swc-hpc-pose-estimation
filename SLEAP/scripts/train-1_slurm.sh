#!/bin/bash

#SBATCH -J slp_train # job name
#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 12G # memory pool for all cores
#SBATCH -n 4 # number of cores
#SBATCH -t 0-06:00 # time (D-HH:MM)
#SBATCH --gres gpu:1 # request 1 GPU (of any kind)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@domain.com 

# Load the SLEAP module
module load SLEAP

# Define directories for SLEAP project and exported training job
SLP_DIR=/ceph/scratch/neuroinformatics-dropoff/SLEAP_HPC_test_data
SLP_JOB_NAME=labels.v001.slp.training_job_2
SLP_JOB_DIR=$SLP_DIR/$SLP_JOB_NAME

# Go to the job directory
cd $SLP_JOB_DIR

# Run the training script generated by SLEAP
./train-script.sh
