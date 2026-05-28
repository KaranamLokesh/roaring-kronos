#!/bin/bash
#
# Run the complete revision-experiment battery on a cloud GPU.
# Self-contained — clones repo, installs deps, runs all experiments,
# packages results.
#
# Tested on Lambda Labs A10 / H100 with the "Lambda Stack" base image.
# Also works on RunPod PyTorch 2.x template.
#
# Expected runtime by GPU tier:
#   H100 SXM5 80GB   ≈ 2.0 hours  →  ~$5
#   A100 80GB        ≈ 3.0 hours  →  ~$5
#   A10 24GB         ≈ 4.5 hours  →  ~$6
#
# Run on the cloud instance:
#   curl -fsSL https://raw.githubusercontent.com/KaranamLokesh/roaring-kronos/main/cloud_benchmark/run_all_experiments.sh | bash
#
# Or interactively (recommended for debugging):
#   git clone https://github.com/KaranamLokesh/roaring-kronos.git
#   cd roaring-kronos
#   bash cloud_benchmark/run_all_experiments.sh

set -e
START_TIME=$(date +%s)
LOG=/tmp/run_all_experiments.log

echo "════════════════════════════════════════════════════════════════"
echo "  Roaring-Kronos full revision experiment battery"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Logging to: $LOG"

# Send all output to both terminal and log file
exec > >(tee -a "$LOG") 2>&1

# ── 1. Setup environment ─────────────────────────────────────────────────────
echo "════════ [SETUP 1/4] Clone repo ════════"

if [ ! -d "roaring-kronos" ]; then
    git clone https://github.com/KaranamLokesh/roaring-kronos.git
fi
cd roaring-kronos
git pull

if [ ! -d "kronos_src" ]; then
    git clone https://github.com/shiyu-coder/Kronos.git kronos_src
fi

echo ""
echo "════════ [SETUP 2/4] Install pip if missing ════════"
if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "Installing python3-pip…"
    sudo apt update -qq && sudo apt install -y -qq python3-pip
fi

echo ""
echo "════════ [SETUP 3/4] Install Python packages ════════"

# Detect CUDA driver and select correct torch wheel
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
    if [ "$CUDA_MAJOR" = "12" ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    elif [ "$CUDA_MAJOR" = "11" ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    fi
    echo "Detected CUDA $CUDA_MAJOR → using torch from $TORCH_INDEX"
else
    echo "WARNING: no nvidia-smi — installing CPU torch (will be slow)"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# Ensure CUDA-enabled torch is installed (force reinstall if CPU torch was there)
python3 -m pip install --quiet --upgrade pip
EXISTING_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
if [[ "$EXISTING_TORCH" == "missing" ]] || \
   python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    : # torch is fine
else
    echo "Existing torch is CPU-only, reinstalling with CUDA support…"
    python3 -m pip uninstall -y torch
    python3 -m pip cache purge >/dev/null 2>&1 || true
fi
python3 -m pip install --quiet torch --index-url "$TORCH_INDEX"
python3 -m pip install --quiet transformers pyroaring pandas numpy huggingface_hub \
    tqdm matplotlib scipy einops safetensors yfinance

# Verify CUDA
if command -v nvidia-smi >/dev/null 2>&1; then
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not reachable from torch'; print(f'  ✓ CUDA ready: {torch.cuda.get_device_name(0)}')"
fi

echo ""
echo "════════ [SETUP 4/4] Verify cached data ════════"
if [ ! -f data/btc_1h.csv ] || [ ! -f data/btc_1h_full_tokens.npy ]; then
    echo "Regenerating BTC tokens (one-time, ~3 min)…"
    PYTHONPATH=. python3 tokenize/explore_tokens.py
else
    echo "  ✓ Token data present"
fi
if [ ! -f data/bitmaps/btc_1h_bitmaps.pkl ]; then
    echo "Regenerating bitmaps…"
    PYTHONPATH=. python3 bitmaps/build_posting_lists.py
else
    echo "  ✓ Bitmaps present"
fi
echo ""

mkdir -p experiments/results cloud_benchmark/results

# ── 2. Run all experiments ───────────────────────────────────────────────────
exp_start() {
    local name=$1
    echo ""
    echo "════════ EXP: $name ════════"
    echo "Start: $(date '+%H:%M:%S')"
}

exp_end() {
    local name=$1
    echo "End:   $(date '+%H:%M:%S')  ✓ $name complete"
}

# Wrap each call so a single failure doesn't abort the whole battery
run_exp() {
    local name=$1
    shift
    exp_start "$name"
    if "$@"; then
        exp_end "$name"
    else
        echo "✗ $name FAILED — continuing with remaining experiments"
    fi
}

run_exp "01 multi_asset_diagnostic" \
    bash -c "PYTHONPATH=. python3 experiments/multi_asset_diagnostic.py"

run_exp "02 cooc_cache_inference" \
    bash -c "PYTHONPATH=. python3 experiments/cooc_cache_inference.py"

run_exp "03 shock_frac_sweep" \
    bash -c "PYTHONPATH=. python3 experiments/shock_frac_sweep.py"

run_exp "04 multi_seed_runner" \
    bash -c "PYTHONPATH=. python3 experiments/multi_seed_runner.py"

run_exp "05 epoch_sweep" \
    bash -c "PYTHONPATH=. python3 experiments/epoch_sweep.py"

# H100/A100 timing benchmark on this GPU
run_exp "06 hw_timing (kronos-small)" \
    bash -c "PYTHONPATH=. python3 cloud_benchmark/a100_benchmark.py \
        --model NeoQuasar/Kronos-small --batches 100 \
        --output cloud_benchmark/results/hw_kronos_small.json"

run_exp "07 hw_timing (kronos-base)" \
    bash -c "PYTHONPATH=. python3 cloud_benchmark/a100_benchmark.py \
        --model NeoQuasar/Kronos-base --batches 100 \
        --output cloud_benchmark/results/hw_kronos_base.json"

# Walk-forward is opt-in via env var (~3-4 hr extra on H100)
if [ "${RUN_WALK_FORWARD:-0}" = "1" ]; then
    run_exp "08 walk_forward_backtest" \
        bash -c "PYTHONPATH=. python3 experiments/walk_forward_backtest.py \
            --windows 6 --start-year 2020 --epochs 5"
else
    echo ""
    echo "════════ SKIPPING walk_forward_backtest ════════"
    echo "To include, re-run with: RUN_WALK_FORWARD=1 bash $0"
fi

# ── 3. Package results ───────────────────────────────────────────────────────
echo ""
echo "════════ Packaging results ════════"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' || echo "cpu")
PKG="results_${GPU_NAME}_$(date +%Y%m%d_%H%M).tar.gz"

tar -czf "$PKG" \
    experiments/results/ \
    experiments/*.png \
    cloud_benchmark/results/ \
    cloud_benchmark/bench.png 2>/dev/null || true

echo "✓ Results packaged: $PKG"
ls -lh "$PKG"

# ── 4. Final report ──────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo "  Elapsed: ${HOURS}h ${MINUTES}m"
echo "  Results: $PKG"
echo ""
echo "  To pull results back to your laptop:"
echo "    scp ubuntu@<this_instance_ip>:$(pwd)/$PKG ."
echo ""
echo "  Then on your laptop:"
echo "    cd ~/Desktop/Misc-projects/roaring-kronos"
echo "    tar -xzf ~/Downloads/$PKG"
echo "    git add experiments/ cloud_benchmark/"
echo "    git commit -m 'Add cloud GPU experiment results'"
echo "════════════════════════════════════════════════════════════════"
