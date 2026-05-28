# Revision Experiments

Four experiments that address the limitations listed in §6 of the paper. Each is runnable on a MacBook with Apple Silicon (MPS) using the existing pyenv 3.10.14 environment. All experiments cache intermediate results and can be safely re-run.

## Quick start

```bash
chmod +x experiments/run_revision_experiments.sh
bash experiments/run_revision_experiments.sh
```

Or run individually below.

---

## 1. Multi-asset replication (≈30 min)

Tests whether the BTC findings (extreme vocabulary sparsity, 52/163 deterministic-s2 coarse tokens) generalize to other assets.

```bash
python experiments/multi_asset_diagnostic.py
```

**Tickers:** BTC-USD (baseline), ETH-USD (alt crypto), SPY (US equity).

**Outputs:**
- `experiments/results/multi_asset_diagnostic.json`
- `experiments/multi_asset_diagnostic.png`

**Paper impact:** Updates Table 1 (vocab utilization) and Table 3 (cooc diagnostic) with ETH and SPY columns. Directly addresses the "BTC-specific?" limitation.

---

## 2. cooc-cache inference optimization (≈15 min)

Implements the inference-time optimization suggested at the end of §4.6: skip the s2 head softmax for deterministic s1 tokens.

```bash
python experiments/cooc_cache_inference.py
```

**Mechanism:** Build a lookup table `s1 → s2` for the 52/163 active s1 values with |Cooc(s1)| = 1. At inference, check the table first; fall back to standard sampling otherwise.

**Outputs:**
- `experiments/results/cooc_cache_inference.json`
- `experiments/cooc_cache_inference.png`

**Paper impact:** Adds a concrete inference-time speedup measurement to §4.6 and Conclusion, transforming the diagnostic from "interesting observation" to "actionable engineering win."

---

## 3. shock_frac sweep (≈2.5 hours)

Sweeps the shock-anchoring fraction `f` over {0.0, 0.15, 0.30, 0.45, 0.60} to find the optimum and characterize sensitivity.

```bash
python experiments/shock_frac_sweep.py
```

**Setup:** Each value runs a full fine-tune (5 epochs) and backtest with seed=42.

**Outputs:**
- `experiments/results/shock_frac_sweep.json`
- `experiments/shock_frac_sweep.png`

**Paper impact:** New figure + table in §4.5 showing how f trades off batch quality vs backtest performance. Addresses "Unswept shock fraction" limitation.

---

## 4. Multi-seed runner (≈3.5 hours)

Runs 3 seeds × 2 samplers (vanilla, roaring) = 6 full fine-tune + backtest combinations. Reports mean ± std across seeds.

```bash
python experiments/multi_seed_runner.py
```

**Seeds:** 42, 137, 2026.

**Outputs:**
- `outputs/models/<sampler>_seed<S>_finetuned/` × 6
- `experiments/results/multi_seed_summary.json`
- `experiments/multi_seed_summary.png`

**Paper impact:** Replaces Table 4 single-point numbers with mean ± std. Addresses the #1 reviewer concern about statistical significance.

---

---

## 5. Epoch sweep (≈3 hours)

Addresses the "Short fine-tune (5 epochs)" limitation. Trains both
samplers for 5/10/20/30 epochs and reports val-loss + backtest
metrics at each length.

```bash
python experiments/epoch_sweep.py
```

**Outputs:**
- `experiments/results/epoch_sweep.json`
- `experiments/epoch_sweep.png`

**Paper impact:** New convergence-curve figure. Determines if Roaring's
slight aggregate disadvantage at 5 epochs closes, persists, or flips
with longer training.

---

## 6. Walk-forward backtest (~16 hours, OPTIONAL)

Addresses the "Single regime test period" limitation by sliding a
12-month-train / 3-month-test window across an extended BTC history
(2020-2026 via Binance free API).

```bash
python experiments/walk_forward_backtest.py --windows 6 --start-year 2020
```

**Outputs:**
- `data/btc_extended_1h.csv` (one-time ~50 MB download)
- `experiments/results/walk_forward.json`
- `experiments/walk_forward_backtest.png`

**Paper impact:** Tests whether the April-2026 win was a regime-specific
fluke or a reproducible pattern across multiple bull/bear/sideways
regimes. The heaviest experiment but the most defensible result.

---

## H100 hardware benchmark (~$1.50 cloud)

Addresses the "Single hardware tier (A10 only)" limitation. See the
runbook at `cloud_benchmark/h100_benchmark.md`. Same script as A10;
just runs on H100 cloud instance.

---

## How to use the results in the paper

After running, the JSONs contain everything you need to update specific tables and figures:

| Paper element | Source |
|---------------|--------|
| Table 1 (vocab utilization) | `multi_asset_diagnostic.json` adds ETH/SPY columns |
| Table 3 (cooc diagnostic) | `multi_asset_diagnostic.json` adds ETH/SPY rows |
| §4.6 implication paragraph | `cooc_cache_inference.json` provides measured speedup |
| §4.5 + new shock_frac figure | `shock_frac_sweep.json` adds sensitivity panel |
| Table 4 (headline backtest) | `multi_seed_summary.json` adds ± std |
| New convergence figure | `epoch_sweep.json` |
| New walk-forward figure | `walk_forward.json` |
| §4.7 update (multi-hw) | H100 benchmark JSON |
| §6 Limitations | Cross out 6 of 7 — frozen tokenizer becomes §7 Future Work |

## Cost estimate

| Step | Time | Cost (cloud A10 @ $1.29/hr) |
|------|------|----------------------------|
| Multi-asset diagnostic | 30 min | $0.65 |
| cooc-cache inference | 15 min | $0.32 |
| shock_frac sweep | 2.5 h | $3.23 |
| Multi-seed runner | 3.5 h | $4.52 |
| Epoch sweep | 3 h | $3.87 |
| H100 benchmark | 30 min @ $2.49 | $1.25 |
| Walk-forward (optional) | 16 h | $20.64 |
| **Total (core 6)** | **~10 h** | **~$14** |
| **+ walk-forward** | **~26 h** | **~$34** |

On MacBook MPS the wall-clock for the core six (minus H100, which
needs CUDA) is similar but cost is zero. Walk-forward at 16 hours is
best done on cloud — the wall-clock penalty for MacBook is significant
because the 6 windows each require fresh fine-tunes.

## Limitations status after all six

| # | Limitation | Status |
|---|------------|--------|
| 1 | Single asset (BTC) | ✅ multi_asset_diagnostic.py |
| 2 | Single seed | ✅ multi_seed_runner.py |
| 3 | Short fine-tune (5 epochs) | ✅ epoch_sweep.py |
| 4 | Single regime test period | ✅ walk_forward_backtest.py |
| 5 | Frozen tokenizer | 🔄 Moved to §7 Future Work in paper |
| 6 | Single hardware tier | ✅ h100_benchmark.md (runbook) |
| 7 | Unswept shock fraction | ✅ shock_frac_sweep.py |

Limitation #5 ("Frozen tokenizer") is fundamentally a separate paper
(would require re-pretraining a BSQ tokenizer on financial data with
Roaring-augmented stage-1 sampling). The current paper now frames it
as the principal direction for follow-up work in §7.
