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

## How to use the results in the paper

After running, the JSONs contain everything you need to update specific tables and figures:

| Paper element | Source |
|---------------|--------|
| Table 1 (vocab utilization) | `multi_asset_diagnostic.json` adds ETH/SPY columns |
| Table 3 (cooc diagnostic) | `multi_asset_diagnostic.json` adds ETH/SPY rows |
| §4.6 implication paragraph | `cooc_cache_inference.json` provides measured speedup |
| §4.5 + new shock_frac figure | `shock_frac_sweep.json` adds sensitivity panel |
| Table 4 (headline backtest) | `multi_seed_summary.json` adds ± std |
| §6 Limitations | Cross out 4 of 7 limitations |

## Cost estimate

| Step | Time | Cost (cloud A10 @ $1.29/hr) |
|------|------|----------------------------|
| Multi-asset diagnostic | 30 min | $0.65 |
| cooc-cache inference | 15 min | $0.32 |
| shock_frac sweep | 2.5 h | $3.23 |
| Multi-seed runner | 3.5 h | $4.52 |
| **Total** | **6.75 h** | **$8.72** |

On MacBook MPS the wall-clock is similar but cost is zero (overnight run).
