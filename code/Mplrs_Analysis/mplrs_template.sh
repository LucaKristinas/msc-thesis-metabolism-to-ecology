#!/bin/bash

#SBATCH --job-name=JOBNAME
#SBATCH --clusters=arc
#SBATCH --qos=priority
#SBATCH --account=path-gut-biomes
#SBATCH --time=TIME 
#SBATCH --nodes=NODES
#SBATCH --ntasks-per-node=TASKS 
#SBATCH --partition=PARTITION
#SBATCH --output=/path/to/logs/NAME.log  
#SBATCH --error=/path/to/errorfiles/NAME.log    

# Load necessary modules
module load OpenMPI/5.0.3-GCC-13.3.0
module load GCC/13.3.0
module load GMP/6.3.0-GCCcore-13.2.0

# Run mplrs with input and output files using mpirun
mpirun -np CORES path/to/lrslib-073/mplrs INE_FILE OUT_FILE -checkp CHECKPOINT_FILE -time 00000 