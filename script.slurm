#!/bin/bash
## Partition
#SBATCH -p hf
## Number of nodes
#SBATCH -N 1  
## Job name
#SBATCH -J mcphonon
## Number of tasks
#SBATCH -n 1
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bruno.hartmann-da-silva@univ-lorraine.fr

module purge                # closing all open modules
module load anaconda/3/5.1  # opening the most recent anaconda

## Creating temporary directory in $SCRATCHDIR
WORKDIR=$SCRATCHDIR/job.$SLURM_JOB_ID.$USER
mkdir -p $WORKDIR

## Copying files to $WORKDIR 
cp -rf '<FILES>'  $WORKDIR

## Moving to $WORKDIR
cd $WORKDIR
## Calculation
python main_program.py '<INPUTS>'

## Copying back files to submit directory
cp -rf * $SLURM_SUBMIT_DIR/.