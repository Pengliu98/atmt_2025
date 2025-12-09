#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --output=out_final_timing.out

echo "Installing packages..."
python3 -m pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install --break-system-packages sentencepiece tqdm sacrebleu

echo ""
echo "Testing environment..."
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python3 -c "import tqdm; import sacrebleu; import sentencepiece; print('All packages OK!')"

echo ""
echo "==================================="
echo "Quick Timing Test for Optimization"
echo "==================================="

cd /home/lipeng/data/atmt_2025

# Create test file
head -50 ~/shares/cz-en/data/raw/test.cz > test_50.cz

# Test 1: Greedy
echo ""
echo "[1/3] Testing Greedy Decoding (k=1)..."
time python3 translate.py \
    --cuda \
    --input test_50.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output test_50_greedy.txt \
    --max-len 300 2>&1 | tail -5

# Test 2: Beam k=3
echo ""
echo "[2/3] Testing Beam Search (k=3)..."
time python3 translate.py \
    --cuda \
    --input test_50.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output test_50_beam3.txt \
    --max-len 300 \
    --beam-size 3 2>&1 | tail -5

# Test 3: Beam k=5
echo ""
echo "[3/3] Testing Beam Search (k=5)..."
time python3 translate.py \
    --cuda \
    --input test_50.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output test_50_beam5.txt \
    --max-len 300 \
    --beam-size 5 2>&1 | tail -5

echo ""
echo "==================================="
echo "RESULTS:"
echo "==================================="
ls -lh test_50*.txt
echo ""
echo "Line counts:"
wc -l test_50*.txt

echo ""
echo "Done!"
