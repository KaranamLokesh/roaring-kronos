#!/bin/bash
#
# Run all four revision experiments in sequence.
# Total wall-clock on MacBook MPS: ~6 hours.
#
# Each experiment can be re-run safely; results are cached and the
# scripts skip work that's already complete.
#
# Output:
#   experiments/results/multi_asset_diagnostic.json
#   experiments/results/multi_seed_summary.json
#   experiments/results/shock_frac_sweep.json
#   experiments/results/cooc_cache_inference.json
#   experiments/*.png  (5 figures)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYENV_VERSION=3.10.14

echo "════════════════════════════════════════════════════════════════"
echo "  Roaring-Kronos revision experiment battery"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Starts: $(date)"
echo "Will run 4 experiments in sequence."
echo ""

# 1. Multi-asset diagnostic — cheapest, do first as a sanity check
echo "════════ [1/4] Multi-asset diagnostic (≈30 min) ════════"
python experiments/multi_asset_diagnostic.py
echo ""

# 2. cooc-cache inference — fast, no fine-tuning
echo "════════ [2/4] cooc-cache inference (≈15 min) ════════"
python experiments/cooc_cache_inference.py
echo ""

# 3. shock_frac sweep — 5 fine-tunes + 5 backtests (≈2.5 hours)
echo "════════ [3/4] shock_frac sweep (≈2.5 hours) ════════"
python experiments/shock_frac_sweep.py
echo ""

# 4. Multi-seed runner — 6 fine-tunes + 6 backtests (≈3.5 hours)
echo "════════ [4/6] Multi-seed runner (≈3.5 hours) ════════"
python experiments/multi_seed_runner.py
echo ""

# 5. Epoch sweep — 8 fine-tunes + 8 backtests (≈3 hours)
echo "════════ [5/6] Epoch sweep (≈3 hours) ════════"
python experiments/epoch_sweep.py
echo ""

# 6. Walk-forward backtest — opt-in (long, requires Binance download)
echo "════════ [6/6] Walk-forward backtest (~16 hours, OPTIONAL) ════════"
echo "Skipping by default. To run, execute:"
echo "    python experiments/walk_forward_backtest.py --windows 6 --start-year 2020"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "All experiments complete."
echo "Ends: $(date)"
echo ""
echo "Results:"
echo "  experiments/results/multi_asset_diagnostic.json"
echo "  experiments/results/cooc_cache_inference.json"
echo "  experiments/results/shock_frac_sweep.json"
echo "  experiments/results/multi_seed_summary.json"
echo ""
echo "Figures:"
echo "  experiments/multi_asset_diagnostic.png"
echo "  experiments/cooc_cache_inference.png"
echo "  experiments/shock_frac_sweep.png"
echo "  experiments/multi_seed_summary.png"
echo "════════════════════════════════════════════════════════════════"
