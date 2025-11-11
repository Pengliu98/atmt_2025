#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=out_averaging.out

module load mamba
source activate atmt

python average_checkpoint.py \
    --checkpoint-dir cz-en/checkpoints/continued/ \
    --number-of-checkpoints 4 \
    --output-checkpoint cz-en/checkpoints/checkpoint_averaged.pt \
    --pattern "checkpoint[0-9]*_*.pt"
