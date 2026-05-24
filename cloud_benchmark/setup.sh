#!/bin/bash
# Setup script for cloud A100 benchmark
# Tested on: Lambda Labs A100 (Ubuntu 22.04, CUDA 12.1)
# Expected runtime: 5-10 minutes setup + 3-5 minutes benchmark

set -e

REPO="https://github.com/KaranamLokesh/roaring-kronos.git"
KRONOS="https://github.com/shiyu-coder/Kronos.git"

echo "════════════════════════════════════════════════════════"
echo "  Cloud A100 Benchmark Setup"
echo "════════════════════════════════════════════════════════"

# 1. Clone the repo if not already here
if [ ! -d "roaring-kronos" ]; then
    echo "[1/5] Cloning roaring-kronos…"
    git clone "$REPO"
fi
cd roaring-kronos

# 2. Clone Kronos source if not vendored
if [ ! -d "kronos_src" ]; then
    echo "[2/5] Cloning Kronos source…"
    git clone "$KRONOS" kronos_src
fi

# 3. Python deps — use python3 -m pip for portability (Lambda base image)
echo "[3/5] Installing Python deps…"
PIP="python3 -m pip"
$PIP install --quiet --upgrade pip
$PIP install --quiet torch transformers pyroaring pandas numpy huggingface_hub \
    tqdm matplotlib scipy einops safetensors yfinance

# 4. Verify data is present (committed in the repo)
if [ ! -f "data/btc_1h_full_tokens.npy" ]; then
    echo "[4/5] Data missing — regenerating tokens…"
    PYTHONPATH=. python tokenize/explore_tokens.py
else
    echo "[4/5] Data already present (btc_1h_*.npy)"
fi

# 5. Verify bitmaps are present
if [ ! -f "data/bitmaps/btc_1h_bitmaps.pkl" ]; then
    echo "[5/5] Bitmaps missing — regenerating…"
    PYTHONPATH=. python bitmaps/build_posting_lists.py
else
    echo "[5/5] Bitmaps already present"
fi

echo ""
echo "Setup complete. Run the benchmark with:"
echo "  cd roaring-kronos"
echo "  python cloud_benchmark/a100_benchmark.py --model NeoQuasar/Kronos-base --batches 100"
echo ""
echo "For Kronos-large (more representative of real pretraining):"
echo "  python cloud_benchmark/a100_benchmark.py --model NeoQuasar/Kronos-large --batches 50"
