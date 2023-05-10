# How to use the SLEAP module
*On the SWC HPC cluster*

> **Abbreviations**
> - SLEAP: Social LEAP Estimates Animal Poses
> - SWC: Sainsbury Wellcome Centre
> - HPC: High Performance Computing

## Aim
This guide explains how to test and use the [SLEAP](https://sleap.ai/) module that is installed on the SWC's HPC cluster for running training and/or inference jobs.

## Prerequisites
* A working knowledge of basic commands on the Linux command line
* An SWC user account
* A local SLEAP installation on PC/laptop (see [below](#install-sleap-on-your-local-pclaptop))
* SWC filesystem mounted on local PC/laptop (see [below](#mount-the-swc-filesystem-on-your-local-pclaptop))

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

## Preparations

### Verify that the SLEAP module is working for you
1. Log into the HPC login node (typing your `<SWC-PASSWORD>` both times when prompted):
   ```bash
   $ ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
   $ ssh hpc-gw1
   ```
2. SLEAP should be listed as one of the available modules:
   ```bash
   $ module avail -l
   SLEAP/2023-03-13
   ```  

3. Start an interactive job on a GPU node:
   ```bash
   $ srun -p gpu --gres=gpu:1 --pty bash -i
   ```
4. Load the SLEAP module. This might take some seconds, but it should finish without errors. Your terminal prompt may change as a result.
   ```
   <SWC-USERNAME>@gpu-350-04:~$ module load SLEAP
   (sleap) <SWC-USERNAME>@gpu-350-04:~$
   ```
   The hostname (the part between "@" and ":") will vary depending on which GPU node you were assigned to.

   To verify that the module was loaded successfully:
   ```bash
   $ module list
   Currently Loaded Modulefiles:
    1) SLEAP/2023-03-13
   ```

   The module is essentially a centrally installed conda environment. When it is loaded, you should be using particular executables for conda and Python. You can verify this by running:

   ```bash
    $ which conda
    /ceph/apps/ubuntu-20/packages/SLEAP/2023-03-13/condabin/conda

    $ which python
    /nfs/nhome/live/<SWC-USERNAME>/.conda/envs/sleap/bin/python
    ```
5. Finally we will verify that the `sleap` python package can be imported and can "see" the GPU. We will just follow the instructions. First, start a Python interpreter:
   ```bash
   $ python
   ```
   Next, run the following Python commands (shown below with their expected outputs):
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
   > The `import sleap` command may take some time to run (more than a minute). This is normal, and it should only happen the first time you import the module. Subsequent imports should be much faster.

6. If all is as expected, you can exit the Python interpreter, and then exit the GPU node
   ```python
    >>> exit()
    $ exit
    ```
   To completely exit the HPC cluster, you will need to logout of the SSH session twice:
   ```bash
   $ logout
   $ logout
   ```

### Install SLEAP on your local PC/laptop
While you can delegate the GPU-intensive work to the HPC cluster, you will still need to do some steps, such as labelling frames, on the SLEAP graphical user interface. For this, you will need to install SLEAP on your local PC/laptop.

We recommend following the official [SLEAP installation guide](https://sleap.ai/installation.html). To be on the safe side, ensure that the local installation is the same version as the one on the cluster - version `1.2.9`.

### Mount the SWC filesystem on your local PC/laptop
The rest of this guide assumes that you have mounted the SWC filesystem on your local PC/laptop. If you have not done so, please follow the relevant instructions on the [SWC internal wiki](https://wiki.ucl.ac.uk/display/SSC/SWC+Storage+Platform+Overview).

We will also assume that the data you are working with are stored in a `ceph` or `winstor` directory to which you have access to. In the rest of this guide, we will use the path `/ceph/scratch/neuroinformatics-dropoff/SLEAP_HPC_test_data` which contains a SLEAP project for test purposes. You should replace this with the path to your own data.

## Training a model
This will consist of two parts - the first part will be done on your local PC/laptop, and the second part will be done on the HPC cluster.

### Part 1: Preparing the data
