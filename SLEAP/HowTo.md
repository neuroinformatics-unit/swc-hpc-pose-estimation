# How to use the SLEAP module
*On the SWC HPC cluster*

## Introduction
This guide explains...

## Prerequisites
* A working knowledge of the Linux command line
* An SWC user account
* Being connected to the SWC network (either on site or via VPN)
* A local SLEAP installation on PC/laptop (see [SLEAP installation guide](https://sleap.ai/installation.html))
* SWC filesystem (`ceph`/`winstor`) on local PC/laptop

## Getting started

### Connecting to the SWC HPC cluster
1. Log into the HPC login node (typing your `<SWC-PASSWORD>` both times when prompted):
   ```bash
   ssh <SWC-USERNAME>@ssh.swc.ucl.ac.uk
   ssh hpc-gw1
   ```
2. List the available modules by typing `module avail -l`. `SLEAP/2023-03-13` should be listed.

3. Load the SLEAP module via `module load SLEAP`. You should get no errors, and your terminal prompt should change to :
   ```
   (sleap) <SWC-USERNAME>@hpc-gw1:~$
   ```