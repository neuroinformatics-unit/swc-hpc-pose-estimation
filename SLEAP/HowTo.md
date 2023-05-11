# How to use the SLEAP module
This guide explains how to test and use the [SLEAP](https://sleap.ai/) module that is installed on the SWC's HPC cluster for running training and/or inference jobs.


> **Abbreviations**
> - SLEAP: Social LEAP Estimates Animal Poses
> - SWC: Sainsbury Wellcome Centre
> - HPC: High Performance Computin
> - SLURM: Simple Linux Utility for Resource Management

> **Note**
> 
> In this document, shell commands will be shown in code blocks like this:
> ```bash
> $ echo "Hello world!"
> ```
> The `$` is not part of the command, it is just a placeholder for the shell prompt. You should not type it when entering the command.
> 
> Similarly, Python commands will be prepended with the `>>>` prompt:
> ```python
> >>> print("Hello world!")
> ```
> The expected outputs of both shell and Python commands will be shown without any prompt:
> ```python
> >>> print("Hello world!")
> Hello world!
> ```

## Prerequisites

### Verify access to the HPC Cluster and the SLEAP module
- Log into the HPC login node (typing your `<SWC-PASSWORD>` both times when prompted):
  ```bash
  $ ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
  $ ssh hpc-gw1
  ```
- SLEAP should be listed as one of the available modules:
  ```bash
  $ module avail -l
  SLEAP/2023-03-13
  ```  
- Start an interactive job on a GPU node:
  ```bash
  $ srun -p gpu --gres=gpu:1 --pty bash -i
  ```
- Load the SLEAP module. This might take some seconds, but it should finish without errors. Your terminal prompt may change as a result.
  ```
  <SWC-USERNAME>@gpu-350-04:~$ module load SLEAP
  (sleap) <SWC-USERNAME>@gpu-350-04:~$
  ```
  The hostname (the part between "@" and ":") will vary depending on which GPUnode   you were assigned to.
  To verify that the module was loaded successfully:
  ```bash
  $ module list
  Currently Loaded Modulefiles:
   1) SLEAP/2023-03-13
  ```
  The module is essentially a centrally installed conda environment. When it   isloaded, you should be using particular executables for conda and Python. You   canverify this by running:
  ```bash
   $ which conda
   /ceph/apps/ubuntu-20/packages/SLEAP/2023-03-13/condabin/conda

   $ which python
   /nfs/nhome/live/<SWC-USERNAME>/.conda/envs/sleap/bin/python
  ```
- Finally we will verify that the `sleap` python package can be imported and can "see" the GPU. We will just follow the [relevant SLEAP instructions](https://sleap.ai/installation.html#testing-that-things-are-working). First, start a Python interpreter:
  ```bash
  $ python
  ```
  Next, run the following Python commands (shown below with their expected outputs  :
  ```python
  >>> import sleap

  >>> sleap.versions()
  SLEAP: 1.2.9
  TensorFlow: 2.6.3
  Numpy: 1.19.5
  Python: 3.7.12
  OS: Linux-5.4.0-139-generic-x86_64-with-debian-bullseye-sid

  >>> sleap.system_summary()
  GPUs: 1/1 available
    Device: /physical_device:GPU:0
           Available: True
          Initalized: False
       Memory growth: None 

  >>> import tensorflow as tf 

  >>> print(tf.config.list_physical_devices('GPU'))
  [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
  ```
  > **Warning**
  > 
  > The `import sleap` command may take some time to run (more than a minute).  This is normal, and it should only happen the first time you import the module.  Subsequent imports should be much faster.

- If all is as expected, you can exit the Python interpreter, and then exit the GPU node
  ```python
   >>> exit()
   $ exit
   ```
  To completely exit the HPC cluster, you will need to logout of the SSH session  twice:
  ```bash
  $ logout
  $ logout
  ```

### Install SLEAP on your local PC/laptop
While you can delegate the GPU-intensive work to the HPC cluster, you will still need to do some steps, such as labelling frames, on the SLEAP graphical user interface. Thus, you also need to install SLEAP on your local PC/laptop.

We recommend following the official [SLEAP installation guide](https://sleap.ai/installation.html). To be on the safe side, ensure that the local installation is the same version as the one on the cluster - version `1.2.9`.

### Mount the SWC filesystem on your local PC/laptop
The rest of this guide assumes that you have mounted the SWC filesystem on your local PC/laptop. If you have not done so, please follow the relevant instructions on the [SWC internal wiki](https://wiki.ucl.ac.uk/display/SSC/SWC+Storage+Platform+Overview).

We will also assume that the data you are working with are stored in a `ceph` or `winstor` directory to which you have access to. In the rest of this guide, we will use the path `/ceph/scratch/neuroinformatics-dropoff/SLEAP_HPC_test_data` which contains a SLEAP project for test purposes. You should replace this with the path to your own data.

## Training a model
This will consist of two parts - [preparing a training job](#prepare-the-training-job) (on your local SLEAP installation) and [running a training job](#run-the-training-job) (on the HPC cluster's SLEAP module).

### Prepare the training job
- Follow the SLEAP instructions for [Creating a Project](https://sleap.ai/tutorials/new-project.html) and [Initial Labelling](https://sleap.ai/tutorials/initial-labeling.html). Ensure that the project file (e.g. `labels.v001.slp`) is saved in the mounted SWC filesystem (as opposed to your local filesystem).
- Next, follow the instructions in [Remote Training](https://sleap.ai/guides/remote.html#remote-training), i.e. "Predict" -> "Run Training…" -> "Export Training Job Package…".
   - For selecting the right configuration parameters, see [Configuring Models](https://sleap.ai/guides/choosing-models.html#) and [Troubleshooting Workflows](https://sleap.ai/guides/troubleshooting-workflows.html)
   - Set the "Predict On" parameter to "nothing". Remote training and inference (prediction) are easiest to run separately on the HPC Cluster.
   - If you are working with a top-down camera view, set the "Rotation Min Angle" and "Rotation Max Angle" to -180 and 180 respectively in the "Augmentation" section.
   - Ensure to save the exported training job package (e.g. `labels.v001.slp.training_job.zip`) in the mounted SWC filesystem, ideally in the same directory as the project file.
   - Unzip the training job package. This will create a folder with the same name (minus the `.zip` extension). This folder contains everything needed to run the training job on the HPC cluster.

### Run the training job
- Login to the HPC cluster as described above.
  ```bash
  $ ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
  $ ssh hpc-gw1
  ```
- Navigate to the training job folder (replace with your own path) and list its contents:
  ```bash
  $ cd /ceph/scratch/neuroinformatics-dropoff/SLEAP_HPC_test_data
  $ cd labels.v001.slp.training_job
  $ls -1
  ```
  There should be a `train-script.sh` file created by SLEAP, which already contains the commands to run the training. You can see the contents of the file by running:
  ```bash
  $ cat train-script.sh
  #!/bin/bash
  sleap-train centroid.json labels.v001.pkg.slp
  sleap-train centered_instance.json labels.v001.pkg.slp
  ```
  The precise commands will depend on the model configuration you chose in SLEAP.
- Next we need to create a SLURM batch script, which will schedule the training job on the HPC cluster. Create a new file called `slurm_train_script.sh` (You can do this in the terminal with `nano`/`vim` or in a text editor of your choice on your local PC/laptop). An example is provided below, followed by explanations.
  ```bash
  #!/bin/bash  
  #SBATCH -p gpu # partition
  #SBATCH -N 1   # number of nodes
  #SBATCH --mem 12G # memory pool for all cores
  #SBATCH -n 2 # number of cores
  #SBATCH -t 0-04:00 # time (D-HH:MM)
  #SBATCH --gres gpu:1 # request 1 GPU (of any kind)
  #SBATCH -o slurm.%N.%j.out # write STDOUT
  #SBATCH -e slurm.%N.%j.err # write STDERR
  #SBATCH --mail-type=ALL
  #SBATCH --mail-user=n.sirmpilatze@ucl.ac.uk  

  # Load the SLEAP module
  module load SLEAP 

  # Define directories for data and exported training job
  DATA_DIR=/ceph/scratch/neuroinformatics-dropoff/SLEAP_HPC_test_data
  JOB_DIR=$DATA_DIR/labels.v001.slp.training_job
  # Go to the job directory
  cd $JOB_DIR 

  # Run the training script generated by SLEAP
  ./train-script.sh
  ```
  
    The `#SBATCH` lines are SLURM directives. They specify the resources needed for the   job, such as the number of nodes, CPUs, memory, etc. For more information, see the   [SLURM documentation](https://slurm.schedmd.com/sbatch.html).
  
    - the `-p gpu` and `--gres gpu:1` options ensure that your job will run on a GPU.   If you want to request a specific GPU type, you can do so with the syntax `--gres   gpu:rtx2080:1`. You can view the available GPU types on the [SWC internal wiki]  (https://wiki.ucl.ac.uk/display/SSC/CPU+and+GPU+Platform+architecture)
    - the `--mem` options refers to CPU memory (RAM), not the GPU one. However, the   jobs often contain steps that use the RAM.
    - the `-t` should be your time estimate for how long the job will take. If it's too   short, SLURM will terminate the job before it's over. If it's too long, it may take   some time to be scheduled (depending on resource availability). With time, you will   build experience on how long various jobs take. It's best to start by running small   jobs (e.g. reduce the number of epochs) and scale up gradually.
    - `-o` and `-e` allow you to specify files to which the standard output and error   will be passed. In the above configuration, the filenames will contain the node   name (`%N`) and the job ID (`$j`)
    - The `--mail-type` and `--mail-user` options allow you to get email notifications   about the progress of your job. Currently email notifications are not working on   the SWC HPC cluster, but this might be fixed in the future.
     
    The `module load SLEAP` line loads the SLEAP module, which we checked earlier.
     
    The `cd` line changes the working directory to the training job folder. This is   necessary because the `train-script.sh` file contains relative paths to the model   configuration and the project file.
    
    The `./train-script.sh` line runs the training job (executes the contained commands)





