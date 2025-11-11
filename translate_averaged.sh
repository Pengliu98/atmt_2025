#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --output=out_translate_averaged.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/joint-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/joint-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_averaged.pt \
    --output cz-en/output_averaged.txt \
    --max-len 300
