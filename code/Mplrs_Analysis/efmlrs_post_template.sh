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

module load GCC/13.2.0
module load Miniconda3/23.9.0-0

source activate /home/path1423/miniforge3/envs/efmlrs-env

efmlrs_post -i path/to/mplrs_output -o path/to/efmlrs_post_ouput --info path/to/efmlrs_info_file