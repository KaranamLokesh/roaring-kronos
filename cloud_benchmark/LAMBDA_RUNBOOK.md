# Lambda Labs — Full Experiment Battery Runbook

End-to-end instructions for running every revision experiment on a
single cloud GPU instance.

## Cost & time on each GPU tier

| GPU                  | Cost/hr  | Full battery | Total cost |
|---------------------|----------|--------------|------------|
| H100 SXM5 80GB      | $2.49    | ~2.0 hours   | ~$5        |
| A100 80GB           | $1.79    | ~3.0 hours   | ~$5        |
| A10 24GB            | $1.29    | ~4.5 hours   | ~$6        |

**Recommended: H100.** Highest speed × cost ratio, results in ~2 hours.

Walk-forward backtest is opt-in (`RUN_WALK_FORWARD=1`) and adds
3-4 hours / $7-10.

## Pre-flight (one-time, on your laptop)

Make sure your SSH key is uploaded to Lambda. The earlier mismatch
was caused by `~/.ssh/id_ed25519.pub` not matching the private key.
Run this to print the correct public key:

```bash
ssh-keygen -y -f ~/.ssh/id_ed25519
```

Copy that line into Lambda's SSH key page if you haven't already.

## Step-by-step

### 1. Launch the instance

1. Go to https://lambdalabs.com → Cloud → GPU Instances
2. Pick `gpu_1x_h100_sxm5` (or `gpu_1x_a100_80gb_sxm4`)
3. Choose region with availability
4. Select your SSH key (the one matching `~/.ssh/id_ed25519`)
5. **Choose the "Lambda Stack" image** (preinstalls CUDA + drivers)
6. Click Launch
7. Wait ~60 seconds for the IP to appear

### 2. SSH in

```bash
ssh ubuntu@<INSTANCE_IP>
```

(Lambda Stack image: SSH usually works first try with no key issues.)

### 3. Verify GPU

```bash
nvidia-smi
```

Should show your GPU (NVIDIA H100, A100, etc.) with no errors.

### 4. Kick off the battery

**Standard run (~2 hours on H100):**
```bash
curl -fsSL https://raw.githubusercontent.com/KaranamLokesh/roaring-kronos/main/cloud_benchmark/run_all_experiments.sh | bash
```

**With walk-forward (~5 hours on H100):**
```bash
curl -fsSL https://raw.githubusercontent.com/KaranamLokesh/roaring-kronos/main/cloud_benchmark/run_all_experiments.sh -o run.sh
RUN_WALK_FORWARD=1 bash run.sh
```

The script:
- Clones the repo + Kronos source
- Installs all Python deps (CUDA torch, transformers, pyroaring, etc.)
- Regenerates BTC tokens + bitmaps if missing
- Runs **all 7 experiments** (8 with walk-forward) in sequence
- Logs everything to `/tmp/run_all_experiments.log`
- Packages results into `results_NVIDIA_<gpu>_<timestamp>.tar.gz`

### 5. Resilience features

If anything crashes mid-run:
- Each experiment is independent; failures don't abort the rest
- All experiments cache their checkpoints and results
- Re-running the script picks up where it left off
- Full log preserved in `/tmp/run_all_experiments.log`

### 6. Pull results back

When the script finishes it will print exact `scp` commands. The
short version, from your laptop:

```bash
# Find the tarball name from the script's output, then:
scp ubuntu@<INSTANCE_IP>:~/roaring-kronos/results_*.tar.gz ~/Downloads/
```

### 7. Terminate the instance

**Immediately** after pulling results. Cost meter runs by the second.

Lambda dashboard → Instances → click your instance → Terminate.

### 8. Apply results locally

```bash
cd ~/Desktop/Misc-projects/roaring-kronos
tar -xzf ~/Downloads/results_NVIDIA_*.tar.gz
git status                          # see what got added
git add experiments/ cloud_benchmark/
git commit -m "Add cloud GPU experiment results"
git push
```

## What you'll get back

Inside the tarball:

```
experiments/results/
├── multi_asset_diagnostic.json
├── cooc_cache_inference.json
├── shock_frac_sweep.json
├── multi_seed_summary.json
├── epoch_sweep.json
└── walk_forward.json           (if RUN_WALK_FORWARD=1)

experiments/
├── multi_asset_diagnostic.png
├── cooc_cache_inference.png
├── shock_frac_sweep.png
├── multi_seed_summary.png
├── epoch_sweep.png
└── walk_forward_backtest.png   (if RUN_WALK_FORWARD=1)

cloud_benchmark/results/
├── hw_kronos_small.json        # H100/A100 timing
├── hw_kronos_base.json
└── bench.png
```

Six JSONs + six PNGs (eight + eight with walk-forward). Each maps
directly to a paper element per `experiments/REVISION_EXPERIMENTS.md`.

## Troubleshooting

**"CUDA not available" in torch:**
- Run `nvidia-smi` to confirm the GPU exists
- If `nvidia-smi` works but torch says CUDA=False:
  ```bash
  python3 -m pip uninstall -y torch
  python3 -m pip cache purge
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```

**"Permission denied (publickey)" on SSH:**
- Verify your local pub key matches what Lambda has:
  ```bash
  diff <(ssh-keygen -y -f ~/.ssh/id_ed25519) <(cat ~/.ssh/id_ed25519.pub)
  ```
- If they differ, regenerate the pub: `ssh-keygen -y -f ~/.ssh/id_ed25519 > ~/.ssh/id_ed25519.pub`
- Re-upload to Lambda's SSH key page

**HuggingFace rate-limited:**
- Sign up for a free token at https://huggingface.co/settings/tokens
- Add `export HF_TOKEN=hf_...` at the start of the script

**Instance shows but no GPU:**
- Some Lambda images (notably plain Ubuntu) don't preinstall drivers
- The script handles this by installing `nvidia-driver-535`
- But it requires reboot. If you see this, terminate and re-launch
  with the "Lambda Stack" image instead

## What this does NOT include

- Stage-1 BSQ tokenizer retraining (would require 12B-record corpus +
  distributed training; this is the §7 Future Work direction)
- Any change to paper.tex (you'll integrate the new numbers manually
  using the per-experiment tables they print at the end)
