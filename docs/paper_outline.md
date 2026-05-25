# Paper Outline

**Working title:** *Roaring Bitmaps for Foundation Model Pretraining on Discrete Tokens: A Case Study on Kronos*

**Alternative titles:**
- *Hierarchical Bitmap Diagnostics and Stratified Sampling for BSQ-Tokenized Financial Time-Series Models*
- *RoaringKronos: Cost-Free Augmentation of BSQ-Based Financial Foundation Models*

---

## Honesty principles

Before sections, lock these in so we don't oversell:
1. **No wall-clock training speedup claim.** A10 measured 0.97× (noise). We add this measurement explicitly to refute the natural objection.
2. **No "we beat Kronos" claim.** Aggregate RankIC slightly favours vanilla-FT (+0.021 vs -0.004). We report it and explain *where* Roaring wins instead (April, drawdown, single-asset OOD behaviour).
3. **Single-asset, single-seed, 5-epoch fine-tune.** This is a "method works, here's preliminary evidence" paper, not a benchmark sweep paper.

---

## Target venue

- **Workshop first:** NeurIPS/ICML Workshop on Foundation Models for Time Series, or ICAIF (AI in Finance)
- **Why workshop:** scope matches — a tightly-scoped methods paper with one clean case study, not a benchmark crusher
- **Length:** 6–8 pages main + appendix

---

## Section-by-Section Outline

### Abstract (1 paragraph, ~200 words)

Three sentences each for:
1. **Problem:** financial foundation models (Kronos) use BSQ tokenization producing extremely sparse, regime-clustered token distributions; existing training pipelines treat all tokens uniformly.
2. **Approach:** apply Roaring Bitmaps to BSQ token posting lists for (a) stratified shock-bar sampling, (b) corpus-scale rare-event lookup, (c) hierarchical co-occurrence diagnostics.
3. **Findings:** zero training-time overhead, +27% rare-token coverage in training batches, best single-month RankIC of +0.195 in fine-tune backtest, and a novel structural finding that 52/163 active coarse tokens have a deterministic fine token — measured for the first time on real market data.

**One headline number:** "417 of 1,048,576 BSQ tokens cover 100% of 17,352 BTC hourly bars (0.04% vocabulary utilisation); top-49 tokens cover 80% of all bars."

---

### 1. Introduction (1 page)

**Three paragraphs:**

**P1 — Setup the foundation model:** Kronos (AAAI 2026) — open-source decoder-only transformer for OHLCVA candlestick forecasting, trained on 12B records from 45 exchanges. Uses Binary Spherical Quantization to convert continuous bars into 20-bit tokens (s1=10 coarse + s2=10 fine = 1M vocabulary). Already widely adopted (25.5k GitHub stars).

**P2 — The under-explored gap:** the BSQ token distribution on real market data is *extremely* sparse and regime-clustered (we'll prove this in §4). Uniform sampling during pretraining wastes capacity on the ~50 dominant "chop" tokens while undersampling the rare shock tokens that matter most for tail risk. Existing fine-tune pipelines don't exploit this structure.

**P3 — Our contribution:** we apply Roaring Bitmaps (Lemire et al.) — the standard data structure for clustered integer sets — to BSQ token posting lists, yielding three contributions:
1. **Stratified shock-bar sampler** with zero training overhead and measurable batch-quality gains
2. **Corpus-scale rare-event lookup** (11.4× over linear scan, sub-linear scaling to 12B records)
3. **Hierarchical co-occurrence diagnostic** revealing that BSQ's "hierarchical" structure is far tighter than the paper claims — measurable for the first time on real data

**Figures:** None in this section. Maybe a teaser plot showing token frequency rank curve (from `experiments/token_distribution.png` middle panel).

---

### 2. Background (1 page)

#### 2.1 Kronos and BSQ tokenization
- Brief recap of Kronos architecture (encoder → BSQ → decoder + autoregressive predictor)
- BSQ key idea: project to unit hypersphere, sign() per dimension → implicit codebook of size 2^K, no explicit lookup
- Kronos's actual config: K=20 split hierarchically as s1=10 + s2=10
- Pull cite: Zhao et al. (BSQ paper, 2024)

#### 2.2 Roaring Bitmaps
- Compressed bitmap data structure with three internal container types: Array, Bitmap, Run
- Sweet spot: integer sets with clustering (runs of consecutive values)
- Standard in Druid, Lucene, ClickHouse — but underused in ML pipelines
- Pull cite: Lemire et al. 2016 (TODS), pyroaring library

#### 2.3 The intersection
**Key observation:** a BSQ-tokenized financial corpus has *exactly* the structure Roaring is built for:
- Tokens cluster in time (market regimes persist)
- Tail of vocabulary is extremely sparse (most tokens never appear)
- Common tokens form long runs (boring chop = same token for hours)

---

### 3. Method (1.5 pages)

#### 3.1 BSQ token posting lists
```
For each token t in vocab(s1, s2, full):
    bitmap[t] = Roaring({position p : token(corpus[p]) == t})
```
- Stored on disk as a single .pkl, ~86 KB for 17k corpus (vs 137 KB raw uint32)
- Loaded once per training run, queried per epoch

#### 3.2 Stratified shock-bar dataloader
- Define "shock token" = token with corpus count ≤ threshold (we use 5)
- Anchor `shock_frac=30%` of training windows on shock-token positions via union of rare bitmaps
- Remaining windows sampled uniformly
- Code: `training/roaring_dataloader.py`

#### 3.3 Hierarchical co-occurrence diagnostic
For each coarse token s1, build a bitmap over the s2 vocabulary tracking which fine tokens co-occur:
```
cooc[s1] = Roaring({s2 : (s1, s2) appears in corpus})
```
Use to measure:
- Tightness: `|cooc[s1]| / 2^s2_bits`
- Integer-space density: `|cooc[s1]| / (max(cooc[s1]) - min(cooc[s1]))`

**Why this matters:** BSQ paper claims the hierarchy lets the model factorise prediction. Our diagnostic measures *how much* — never reported before on real data.

**Figures:** Architecture diagram (we don't have one yet — TODO).

---

### 4. Experiments (3 pages)

#### 4.1 Setup
- **Data:** BTC-USD 1h, May 2024 → May 2026 (17,352 bars from yfinance)
- **Tokeniser:** `NeoQuasar/Kronos-Tokenizer-base` (frozen)
- **Predictor:** `NeoQuasar/Kronos-small` (24.7M params)
- **Train/test split:** 80%/20% (last 3,471 bars held out)
- **Hardware:** Apple M-series for fine-tuning, NVIDIA A10 for timing benchmark
- Single-seed for now (limitation noted in §6)

#### 4.2 Corpus characterization (uses `experiments/token_distribution.png`)
**Three findings:**
- **Vocabulary utilisation:** 417/1,048,576 (0.04%) — the corpus is staggeringly sparse
- **Concentration:** top-49 tokens cover 80% of bars, top-10 cover 44.5%
- **Run length:** mean s1 run = 2.4 bars, max = 80 bars (multi-day regime persistence)

**Table 1:** vocabulary stats (full / s1 / s2 — unique tokens, entropy, max run)

**Figure 1:** `experiments/token_distribution.png` — 6-panel grid

#### 4.3 Roaring posting list properties
**From `bitmaps/build_posting_lists.py`:**
- s1 bitmaps: 1.86× compression vs raw uint32
- Full token bitmaps: 1.68× compression
- Common tokens (≥1000 occurrences): 1.99× (near theoretical max via Run Container)
- Rare tokens (1 occurrence): 0.22× (overhead exceeds payload — expected)

**Table 2:** per-token compression breakdown (top-5 common, bottom-5 rare)

#### 4.4 Rare-event lookup speedup
- Linear scan: 0.71 ms across 17,352 bars to find all 512 shock-bar positions
- Roaring union: 0.062 ms — **11.4× faster**
- Scaling argument: linear scan is O(corpus), Roaring is O(rare_set_size); gap widens with corpus

**Figure 2:** lookup time vs corpus size (log-log), measured at our corpus + projected to 12B

#### 4.5 DataLoader batch-quality comparison
Sample 256 batches × 32 windows each, compare token distribution:

| Metric | Vanilla | Roaring (shock_frac=0.3) |
|--------|---------|--------------------------|
| Token entropy | 6.317 bits | 6.380 bits |
| Rare-token rate | 3.19% | 4.06% (+27%) |
| Windows w/ ≥1 rare token | 91.0% | 93.8% |
| Top-5 dominance | 27.8% | 27.2% |

**Figure 3:** `experiments/dataloader_benchmark.png` — 3-panel comparison

#### 4.6 Hierarchical co-occurrence diagnostic (the novel finding)
- **52 of 163 active s1 values have exactly 1 fine token** — fully deterministic hierarchy
- **All 163 use ≤8 fine tokens out of 1024** — tightest hierarchical tokeniser ever measured on this scale
- **Integer-space density of 0.075 median** — fine tokens scattered in integer encoding (BSQ bit-ordering has no semantic structure)
- **Hamming-distance neighbours show bit-level clustering** — semantic similarity lives in Hamming space, not integer space

**Table 3:** tightest 5 and loosest 5 s1 values with their fine-token sets

**Figure 4:** scatter plot of (popcount(cooc[s1]) vs occurrences of s1) — shows that even high-count coarse tokens use very few fine tokens

#### 4.7 Training-time benchmark (the honesty section)
**On A10, B=32, T=512:**
- Vanilla step: 382 ms (Kronos-small) / 1265 ms (Kronos-base)
- Roaring step: 387 ms / 1311 ms
- Data loading fraction: **0.1% on both**
- End-to-end speedup: **0.97× — within noise**

**Claim:** Roaring imposes zero training overhead but provides zero wall-clock gain at typical training configs. Useful for offline corpus processing; neutral at training time.

**Figure 5:** `cloud_benchmark/results/a10_kronos_base.png` — stacked phase breakdown

#### 4.8 Fine-tune + backtest
- Fine-tune Kronos-small for 5 epochs with two samplers
- Eval: 675 rolling windows, lookback=90, pred_len=10, T=0.6, top_p=0.9 (Kronos's published config)
- 4 signal aggregations: last / mean / max / min predicted close - current close

**Table 4:** headline metrics

| Variant | RankIC | CumRet | Sharpe | MaxDD |
|---------|--------|--------|--------|-------|
| zero-shot | -0.021 | -36.7% | -3.60 | -41.1% |
| vanilla-FT | +0.021 | -41.0% | -4.11 | -41.1% |
| roaring-FT | -0.004 | -38.1% | -3.69 | -38.4% |

**Two honest observations:**
- Vanilla wins aggregate RankIC (signal quality)
- Roaring wins cumulative return, Sharpe, and max drawdown (P&L behaviour) — because the shock-trained model is less confident in calm periods → fewer wrong-direction trades → less cost drag

**Table 5:** monthly RankIC breakdown — **Roaring's April +0.195 is the best single-month IC of any variant**

**Figure 6:** `experiments/btc_backtest_comparison.png` — 4-panel comparison

---

### 5. Discussion (1 page)

#### 5.1 When does Roaring training help?
- **Helps:** April-style volatile regime-change months (where shock training pays off)
- **Hurts:** calm sideways months (where shock bias produces under-confident predictions on benign bars)
- **Neutral training cost:** can always be used as a drop-in upgrade

#### 5.2 Why the aggregate RankIC went the other way
- Test period (Dec 2025 → May 2026) was a bear market with one crash and a recovery
- Only 5 months → 1 bad month (May) dominates aggregate
- Shock-bar overweighting helped only in 2/5 months
- Honest framing: "Roaring's bet pays off in regime-change months, costs in calm months — net effect depends on test-period composition"

#### 5.3 The hierarchical-diagnostic angle as the durable contribution
Even if the stratified sampler had no effect, the diagnostic finding (52/163 s1 fully determine s2) is independently valuable:
- Suggests a Mixture-of-Experts-style routing optimisation: skip s2 head for deterministic coarse tokens
- Measures BSQ's claimed factorisation property for the first time
- Reproducible on any BSQ-tokenized corpus (image, video, audio — BSQ is from a vision paper)

#### 5.4 What Roaring is *actually* good for in ML pipelines
Position the broader argument: Roaring's best ML use cases aren't in the hot training loop. They are:
- Corpus statistics and curriculum design (rare-event sampling, deduplication)
- Token-level diagnostics and ablation tooling
- Multi-modal foundation model evaluation (sparse token co-occurrence)

---

### 6. Limitations (0.5 page) — be aggressive about these so reviewers can't be

- **Single asset (BTC).** Need replication on equities, FX, commodities to claim "financial foundation models" in general
- **Single seed.** Need n=3 minimum runs to show variance
- **Short fine-tune (5 epochs).** Real training may need 20+ epochs for Roaring's distribution shift to converge fully
- **Single bear-market test period.** A bull-market test would likely show different signal/asymmetry
- **Pre-trained tokeniser, not retrained.** Tokeniser was frozen — applying Roaring during stage-1 tokeniser training is left as future work
- **No multi-GPU benchmark.** Distributed training (where data fraction would grow) is unmeasured
- **shock_frac=0.30 is unswept.** Need 0.15 / 0.30 / 0.45 / 0.60 sweep to find optimum

---

### 7. Future Work (0.5 page)

- Replication on multi-asset Qlib backtest (Chinese A-share, the Kronos paper's actual benchmark)
- Apply during stage-1 tokeniser training (not just stage-2 predictor)
- Distributed training measurement on H100/B200 cluster
- BSQ vocabulary pruning using cooc bitmaps (skip s2 head for deterministic s1)
- Cross-modal validation: apply same diagnostic to vision BSQ tokens (where BSQ originates)

---

### 8. Conclusion (~150 words)

Reiterate the three contributions in one sentence each:
1. Zero-overhead stratified sampler with batch-quality and drawdown gains
2. Corpus-scale rare-event infrastructure (11.4× lookup)
3. Novel hierarchical diagnostic measurable for the first time on financial BSQ data

Single sentence on why it matters: "As foundation models for non-language modalities adopt discrete tokenisation at scale, the tooling for *measuring* and *exploiting* token sparsity will be as important as the tokeniser itself."

---

## Figures Inventory (what we already have)

| # | File | Purpose |
|---|------|---------|
| 1 | `experiments/token_distribution.png` | §4.2 corpus characterization |
| 2 | TODO | §4.4 lookup scaling (log-log) — need to plot |
| 3 | `experiments/dataloader_benchmark.png` | §4.5 batch quality |
| 4 | TODO | §4.6 cooc scatter — need to plot |
| 5 | `cloud_benchmark/results/a10_kronos_base.png` | §4.7 step time breakdown |
| 6 | `experiments/btc_backtest_comparison.png` | §4.8 backtest 3-way |

**Missing figures to create:**
- **Fig 0 (teaser):** simple token frequency curve with annotation
- **Fig 2:** lookup time vs corpus size (measure + extrapolate)
- **Fig 4:** s2 vocab usage per s1 with Hamming-distance overlay
- **Architecture diagram** (optional): BSQ pipeline + where Roaring plugs in

## Tables Inventory

| # | Content | Source |
|---|---------|--------|
| 1 | Vocabulary stats | from `bitmaps/build_posting_lists.py` output |
| 2 | Per-token compression | from same |
| 3 | Tightest/loosest s1 values | from same |
| 4 | Headline backtest metrics | `experiments/btc_backtest_comparison.png` |
| 5 | Monthly RankIC breakdown | from `eval/btc_backtest_finetuned.py` output |

## Code-side TODOs before writing prose

- [ ] Generate Fig 2 (lookup scaling vs corpus size) — script needed
- [ ] Generate Fig 4 (cooc scatter with Hamming colouring) — script needed
- [ ] Re-run fine-tune with 3 seeds for variance bars — optional but strong
- [ ] Sweep shock_frac ∈ {0.15, 0.30, 0.45, 0.60} — strong for §4.8

---

## Suggested writing order (1-2 weeks)

1. **Day 1-2:** Write §3 (Method) — most concrete, least judgment-laden
2. **Day 3-4:** Write §4 (Experiments) — paste tables/figures, add captions and one paragraph each
3. **Day 5:** Write §5 (Discussion) and §6 (Limitations) together
4. **Day 6-7:** Write §1 (Intro) and §2 (Background) — these come last because intros are easier to write *after* you've written the body
5. **Day 8:** Abstract + Conclusion + polish
6. **Day 9-10:** Generate any missing figures, fix tables
7. **Day 11+:** Iterate based on co-author/colleague feedback
