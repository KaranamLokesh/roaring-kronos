# H100 Benchmark Runbook

Addresses limitation 7: "Single hardware tier (A10 only)."

## Why H100

The A10 measurement showed data loading is~0.1% of step time. The
hypothesis we want to test is whether this fraction **grows** on
faster accelerators — if H100's forward+backward is 3-5× faster, the
relative cost of data loading should rise.

A successful measurement would either:
- **Confirm** the hypothesis: data fraction on H100 ≈ 0.3-0.5%,
  still small, supporting the paper's framing that data-path
  optimisations don't matter at typical single-GPU configs
- **Disconfirm** the hypothesis: data fraction stays at 0.1% or
  shrinks further, in which case we'd update the discussion

Either result tightens the paper.

## What to run

The benchmark script is already general-purpose; it will work on
H100 unchanged:

```bash
# On the H100 instance
git clone https://github.com/KaranamLokesh/roaring-kronos.git
cd roaring-kronos
git clone https://github.com/shiyu-coder/Kronos.git kronos_src
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers pyroaring pandas numpy huggingface_hub \
    tqdm matplotlib scipy einops safetensors yfinance

PYTHONPATH=. python cloud_benchmark/a100_benchmark.py \
    --model NeoQuasar/Kronos-small --batches 100 --output h100_small.json
PYTHONPATH=. python cloud_benchmark/a100_benchmark.py \
    --model NeoQuasar/Kronos-base  --batches 100 --output h100_base.json
```

Total runtime on H100: ~5 minutes for both.

## Provider options (priced May 2026)

| Provider | H100 SXM5 80GB | Notes |
|----------|----------------|-------|
| Lambda Labs | $2.49/hr | Most reliable, often out of stock |
| RunPod | $2.79/hr Secure / $2.39 Community | Best UI, easy SSH |
| Vast.ai | $1.80–2.20/hr (spot) | Cheapest, occasional reliability issues |
| Modal | $3.99/hr | Pay-per-second, no idle billing |

For a 30-minute run including setup, expect total cost ~$1.50.

## Recommended provider for this paper

**RunPod with the "PyTorch 2.x" template.** It comes with CUDA + torch
preinstalled, saves you 5 minutes of setup. Worth the $0.40 premium
over Vast.

```bash
# RunPod web UI:
#   Deploy → Pods → H100 80GB SXM5
#   Template: PyTorch 2.x
#   Click Deploy

# Once running, in the web terminal:
git clone https://github.com/KaranamLokesh/roaring-kronos.git
cd roaring-kronos
git clone https://github.com/shiyu-coder/Kronos.git kronos_src
pip install pyroaring yfinance einops safetensors

PYTHONPATH=. python cloud_benchmark/a100_benchmark.py \
    --model NeoQuasar/Kronos-base --batches 100 --output h100_base.json

# Pull the results back
# (RunPod offers a Files panel; download h100_base.json and bench.png)
```

Total time: ~10 minutes active + ~5 minutes inference = ~$0.50.

## After running

Save the result files into the repo:

```bash
mkdir -p cloud_benchmark/results
mv h100_small.json   cloud_benchmark/results/h100_kronos_small.json
mv h100_base.json    cloud_benchmark/results/h100_kronos_base.json
mv bench.png         cloud_benchmark/results/h100_kronos_base.png
```

Then update `cloud_benchmark/results/ANALYSIS.md` with a new section
comparing A10 → H100 data-loading fractions. The headline finding will
go into the paper's §4.7.

## Expected interpretation

If data fraction is **still small (<1%) on H100**:
> "Even on next-generation accelerators with 3-5× higher arithmetic
> throughput, data loading remains a minor fraction of training
> step time. The Roaring dataloader's CPU-side speedup is essentially
> invisible in single-GPU training; its value remains in offline
> corpus operations and data quality, not training wall-clock."

If data fraction **grows meaningfully on H100 (>5%)**:
> "On H100, data loading rises to X% of step time, suggesting that
> Roaring's measured 1.48× dataloader speedup would yield a
> measurable Y% end-to-end training improvement at this hardware
> tier — a real wall-clock win on frontier compute that was hidden
> at the A10 scale."

Either interpretation is publishable. The honest measurement matters
more than which way it lands.

## What we are NOT doing

We are NOT re-running the fine-tune backtest on H100. The fine-tune
results from MacBook MPS are fully reproducible on any device and the
backtest metrics are device-independent. Only the **per-phase
timing** measurement needs different hardware to be informative.
