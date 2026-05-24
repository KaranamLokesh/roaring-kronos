# Cloud A100 Benchmark

Locks in the *real* data-loading fraction on an A100 GPU — replaces the projected 25% with measured numbers.

## Why this matters

Our `experiments/time_benefits_analysis.py` projects 15 hours saved at Kronos full scale, but assumes the data-loading fraction is 25% on A100. That assumption is the weakest link in the cost claim. This benchmark measures the actual fraction.

## What it measures

Per-step time breakdown across two samplers (vanilla vs Roaring):

| Phase | What it is |
|-------|-----------|
| `data_loading_ms` | CPU work to assemble batch (this is what Roaring optimises) |
| `h2d_transfer_ms` | CPU → GPU memcopy |
| `tokenize_ms` | Frozen tokenizer forward (GPU) |
| `forward_ms` | Predictor model forward |
| `backward_ms` | Loss + backprop |
| `step_ms` | Optimiser step |

Output: `results.json` + `bench.png` with side-by-side breakdown.

## Cost estimate

- **Lambda Labs A100 (80GB):** ~$1.29/hr
- **RunPod A100 (80GB):** ~$1.89/hr
- **Vast.ai A100:** ~$0.80–1.20/hr (spot)

Full benchmark including setup: **~15 minutes ≈ $0.30–0.50**.

## Quick start (Lambda Labs)

```bash
# 1. Launch a 1×A100 instance on Lambda (or any provider with CUDA 12.x)
# 2. SSH in, then:

curl -fsSL https://raw.githubusercontent.com/KaranamLokesh/roaring-kronos/main/cloud_benchmark/setup.sh | bash

# 3. Run the benchmark (Kronos-base is the most representative)
cd roaring-kronos
python cloud_benchmark/a100_benchmark.py --model NeoQuasar/Kronos-base --batches 100

# 4. Pull results back
scp $INSTANCE:roaring-kronos/results.json .
scp $INSTANCE:roaring-kronos/bench.png .

# 5. Tear down the instance — total cost ~$0.50
```

## Args

```
--model        HF id — Kronos-small | Kronos-base | Kronos-large  (default: Kronos-base)
--batches      Number of timed batches  (default: 100)
--batch-size   Per-GPU batch size  (default: 32)
--seq-len      Sequence length  (default: 512, Kronos max_context)
--shock-frac   Roaring shock fraction  (default: 0.30)
--output       JSON output path  (default: results.json)
```

## After running

Pull `results.json` back and update `experiments/time_benefits_analysis.py` to use the measured `data_fraction` instead of the assumed 0.25. The projected savings number becomes a defensible measurement.

## Recommended runs

For a complete paper-ready result, run on three model sizes:

```bash
python cloud_benchmark/a100_benchmark.py --model NeoQuasar/Kronos-small --batches 100 --output small.json
python cloud_benchmark/a100_benchmark.py --model NeoQuasar/Kronos-base  --batches 100 --output base.json
python cloud_benchmark/a100_benchmark.py --model NeoQuasar/Kronos-large --batches 50  --output large.json
```

The data-loading fraction should **grow with model size** — larger models = bigger relative cost of slow data loading. This is the cleanest narrative for the paper.
