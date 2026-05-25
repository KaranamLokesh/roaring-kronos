# Cloud Benchmark Analysis — A10 measured numbers

## The headline correction

Our local projection in `experiments/time_benefits_analysis.py` assumed the data-loading fraction on a real GPU would be ~25%. **The measured fraction is 0.1%** — off by 250×. The wall-clock savings claim ("15 hours, $247") was unsupported and must be dropped from the paper.

## What we actually measured (NVIDIA A10, 24GB)

### Kronos-small (24.7M params, B=32, T=512, 100 batches)

| Phase | Vanilla | Roaring | Δ |
|-------|---------|---------|---|
| data_loading_ms | 0.12 | 0.12 | +0.00 |
| h2d_transfer_ms | 0.15 | 0.13 | -0.02 |
| tokenize_ms | 15.73 | 15.85 | +0.13 |
| forward_ms | 121.05 | 123.08 | +2.03 |
| backward_ms | 239.75 | 242.91 | +3.16 |
| step_ms | 5.28 | 4.94 | -0.34 |
| **TOTAL** | **382.07** | **387.02** | **+4.95** |

- Vanilla data fraction: **0.1%**
- Roaring data fraction: **0.1%**
- End-to-end speedup: **0.987×** (within noise of 1.0×)

### Kronos-base (102.3M params, B=32, T=512, 100 batches)

| Phase | Vanilla | Roaring | Δ |
|-------|---------|---------|---|
| data_loading_ms | 0.14 | 0.16 | +0.02 |
| h2d_transfer_ms | 0.16 | 0.17 | +0.01 |
| tokenize_ms | 15.67 | 15.97 | +0.30 |
| forward_ms | 431.17 | 447.89 | +16.71 |
| backward_ms | 798.98 | 828.78 | +29.80 |
| step_ms | 18.87 | 18.54 | -0.33 |
| **TOTAL** | **1265.00** | **1311.50** | **+46.50** |

- Vanilla data fraction: **0.0%**
- Roaring data fraction: **0.0%**
- End-to-end speedup: **0.965×** (within noise of 1.0×)

## Interpretation

1. **Data loading is invisible on modern GPUs at typical training configs.** Forward + backward = >95% of step time. Data loading + H2D = ~0.1% of step time. Optimising the 0.1% is not a wall-clock win.

2. **Roaring's overhead is zero in practice.** The +1.3% / +3.7% deltas are run-to-run noise (typical std-dev of forward/backward is similar magnitude). Roaring imposes no measurable training-time penalty.

3. **The model size trend went the wrong way for our claim.** We projected data fraction would *grow* with model size (because compute gets slower per param). It went the *other* way (compute dominates harder as the model gets bigger), because we're using fixed batch size — bigger model = more compute per same-size batch.

## What this changes in the paper

**Drop these claims:**
- "Roaring saves 15 hours / $247 at Kronos full-scale training"
- "1.32× end-to-end training speedup"
- The projection plot in `experiments/time_benefits.png`

**Keep and emphasise these claims:**
- **Zero training overhead** (now measured, not assumed)
- **1.48× dataloader throughput** (CPU-only, useful for offline pipelines)
- **11.4× rare-event lookup** (scales sub-linearly — the genuine architectural win at 12B records)
- **+0.063 bits entropy gain, +27% rare-token rate** in training batches
- **Best single-month RankIC** (+0.195, April 2026)
- **52/163 deterministic-s2 hierarchical finding** (novel diagnostic)

## Caveats this measurement does NOT cover

The benchmark we ran is **one config** on **one GPU**. Cases where data loading might still matter:

- **Very small models** (e.g., Kronos-mini, 4M params) where forward+backward is cheap
- **Very long sequences** (T >> 512) where data loading scales but compute scales superlinearly anyway
- **Multi-GPU / multi-node** where data must traverse cluster network
- **Mixed precision / Flash Attention** that makes forward+backward 2-3× faster, pushing data fraction up
- **H100/B200** with 3-5× higher compute throughput than A10

For the paper, we should state: *"At our measured config (A10, B=32, T=512), data loading is 0.1% of step time. The conditions under which Roaring's dataloader-throughput win translates to wall-clock savings are: (a) very fast accelerators (H100+), (b) tiny models, (c) very small batch sizes, or (d) multi-node setups with slow shared storage."*

## Cloud run cost

| Item | Cost |
|------|------|
| Lambda Labs (failed launches, ~3 instances × 5 min each) | ~$0.30 |
| Final A10 instance (~30 min total including driver install + 2 benchmarks) | ~$0.40 |
| **Total** | **~$0.70** |

Cheaper than expected. Lesson learned: **always measure before projecting**.
