# Research Dossier: Roaring Bitmaps + BSQ Tokenizers + Financial Foundation Models

**Scope:** Lit review for ICAIF paper on Roaring Bitmap augmentation of Kronos' BSQ-tokenized financial foundation model pipeline — covering financial time-series foundation models, discrete tokenization (BSQ family), and bitmap-based systems applied to ML.
**Angle:** Strengthening positioning + Related Work section of an existing paper draft (docs/paper.tex)
**Date:** 2026-05-24

---

## 1. Executive Summary

Three observations reshape the paper's positioning:

1. **The BSQ "extreme sparsity" finding is novel for financial data but echoes a known image/video tokenizer concern.** The codebook-collapse literature (CODA, VQBridge, FSQ) is actively trying to *fix* underutilization. We instead *measure* and *exploit* it — a complementary angle.
2. **Hierarchical co-occurrence diagnostic plugs into an active gap.** The "Early Quantization Shrinks Codebook" paper (arXiv 2603.17052) explicitly calls for diagnostic tools for discrete tokenizers. Our diagnostic is the first such tool measured on a financial BSQ model.
3. **Stratified sampling at token level is unexplored for financial foundation models.** S-OHEM (2017) does it at object-proposal level for detection. Curriculum-learning-for-LLM-pretraining (Beyond Random Sampling, 2025) does it at document level. Nobody has done token-level stratified sampling with bitmap-backed shock-event selection for time-series foundation models — that's our gap.

Bottom line: the paper's contribution is **stronger** than the original draft suggested. It bridges three distinct literatures that don't yet talk to each other.

---

## 2. Existing Methods — Taxonomy

### 2.1 Financial Time-Series Foundation Models

**Mechanism in one sentence.** Pretrained transformers that ingest historical price/volume series and forecast future values, distinguished by their tokenization choice and architectural prior.

**Representative works:**
- **Kronos** (Shi et al. 2025, AAAI 2026, arXiv 2508.02739) — decoder-only, BSQ tokenization, 12B candlesticks across 45 exchanges → 93% RankIC improvement over leading TSFM
- **Chronos** (Ansari et al. 2024, arXiv 2403.07815) — T5-family, scaled quantile binning (4096 bins), general time-series across domains
- **Moirai** (Woo et al. 2024) — encoder-only, any-variate attention, no quantization, multi-variate first-class
- **TimesFM** (Das et al. 2024, arXiv 2310.10688, ICML 2024) — decoder-only, continuous patch embeddings, 100B time-points
- **Lag-Llama** (Rasul et al. 2023, arXiv 2310.08278) — decoder-only, probabilistic output via distribution head, uses lags as covariates
- **BloombergGPT** (Wu et al. 2023, arXiv 2303.17564) — 50B parameter text-only LLM for finance; closed
- **FinGPT** (Yang et al. 2023, arXiv 2306.06031) — open-source LoRA-tuned LLM on financial text

**Strengths:** Zero-shot transfer across assets; probabilistic forecasts; consume any time series.

**Weaknesses (relevant to our work):** None of these inspect or exploit the structure of their tokenized corpus. Kronos uses BSQ but doesn't measure utilization. The text-only LLMs (Bloomberg, FinGPT) are different modality and not directly comparable.

### 2.2 Discrete Tokenization (BSQ Family)

**Mechanism in one sentence.** Project an encoder output to a low-dim latent and quantize each dimension to a small set of discrete values, producing a token index that an autoregressive model can predict.

**Representative works:**
- **VQ-VAE** (van den Oord 2017) — explicit codebook; nearest-neighbor lookup; suffers codebook collapse
- **RVQ** (residual VQ; SoundStream, EnCodec) — stack of VQ codebooks on residuals; widely used in audio
- **FSQ** (Mentzer et al. ICLR 2024, arXiv 2309.15505) — replace VQ with element-wise rounding to small finite sets; ~100% utilization by design, no auxiliary losses
- **LFQ / MagViT-v2** (Yu et al. ICLR 2024, arXiv 2310.05737) — drop codebook entirely; element-wise sign on projected features; supports vocab of 2^18
- **BSQ** (Zhao, Xiong, Krähenbühl ICLR 2025, arXiv 2406.07548) — LFQ + unit-sphere normalization for bounded quantization error; hierarchical s1/s2 splitting for tractable entropy; 2.4× throughput vs prior tokenizers, 100× visual compression

**Codebook utilization / diagnostic line:**
- **"Beyond Stationarity"** (arXiv 2602.18896) — rethinks codebook collapse mechanisms
- **VQBridge** (arXiv 2509.10140) — claims 100% codebook utilization with scalable training
- **"Early Quantization Shrinks Codebook"** (arXiv 2603.17052) — proposes "deferred quantization" + a *shrinkage diagnostic suite*. Argues that reconstruction-centric evaluation hides downstream diversity problems.
- **Discrete Tokenization Survey** (arXiv 2507.22920) — comprehensive 2025 survey including BSQ position in the family

**Strengths of BSQ:** Bounded quantization error; no codebook lookup; scalable to large vocabularies; hierarchical structure enables factored prediction.

**Weaknesses (relevant to our work):** BSQ paper benchmarks on image/video reconstruction. **Nobody has measured per-token utilization or hierarchical conditional sparsity on a financial corpus.** Kronos uses BSQ but only reports aggregate cross-entropy.

### 2.3 Bitmap-Based Systems in ML / Data Engineering

**Mechanism in one sentence.** Compressed bitmap data structures (typically Roaring or Bloom-based) enable sub-linear set operations over large integer-indexed populations.

**Representative works:**
- **Roaring Bitmaps** (Lemire et al. 2016, *Software: Practice & Experience*) — hybrid Array/Bitmap/Run containers; standard in Lucene/Druid/Spark/Pinot
- **LSHBloom** (Zelenfreund et al. 2024, arXiv 2411.04257) — extreme-scale document deduplication via MinHash + Bloom filter
- **Dolma** (AI2, 2024) — 3T-token LLM pretraining corpus; uses Bloom + MinHash + LSH for multi-stage dedup
- **Deduplicating Training Data** (Lee et al. ACL 2022) — canonical demonstration that dedup improves LLM training
- **MegaScale** (Jiang et al. 2024, arXiv 2402.15627) — system-side study of training at 10k+ GPUs; documents data-loading bottlenecks
- **SSDTrain** (Wu et al. 2024, arXiv 2408.10013) — activation offloading framework; addresses data path at scale

**Strengths:** Sub-linear set ops; production-ready; widely deployed.

**Weaknesses for ML adoption:** Used heavily in retrieval/dedup but rarely in the training loop itself. GPU forward/backward dominates step time at typical configs (single-GPU, no Flash Attention) — confirmed by our own A10 measurement.

### 2.4 Stratified Sampling and Curriculum Learning

**Mechanism in one sentence.** During training, sample harder/rarer/more-informative examples more often than uniform random sampling would.

**Representative works:**
- **S-OHEM** (Li et al. 2017, arXiv 1705.02233) — Stratified Online Hard Example Mining for object detection; strata by classification/localization loss ratio
- **Curriculum Learning** (Bengio et al. 2009) — canonical formulation; easy-to-hard ordering
- **Importance Sampling for Tail Risks** (arXiv 2307.04676) — finance/simulation classic; importance sampling for rare-event Monte Carlo
- **"Beyond Random Sampling: Curriculum LLM Pretraining"** (2025, arXiv 2506.11300) — 18-45% step reduction via curriculum at pretraining
- **Optimal Pretraining Data Mixtures** (ACL 2025, aclanthology 2025.acl-long.1564) — learn the data mixture sampling rates
- **Hard Sample Mining: Efficient Robust Training** (2025) — survey

**Strengths:** Real gains on rare-class tasks; well-motivated theoretically.

**Weaknesses (relative to our work):** All operate at example-level (entire documents/images/proposals), not at token-level inside a sequence. None use bitmap infrastructure for cheap rare-event selection. None applied to time-series foundation models.

---

## 3. Comparison Table

| Method | Year | Domain | Mechanism | Key claim | Code | Limitation |
|--------|------|--------|-----------|-----------|------|------------|
| Kronos | 2025 | Finance FM | BSQ + autoregressive | +93% RankIC vs prior TSFM | ✓ | Doesn't measure tokenizer structure |
| Chronos | 2024 | TS FM | Quantile binning (4096 bins) | Beats GP / DeepAR on 42 datasets | ✓ | Bin centers not data-aware |
| Moirai | 2024 | TS FM | Encoder + any-variate attn | Strong cross-series transfer | ✓ | No quantization → no token sparsity story |
| TimesFM | 2024 | TS FM | Decoder + continuous patches | Near-SOTA zero-shot | ✓ | Continuous (no token analysis) |
| Lag-Llama | 2023 | TS FM | Decoder + probabilistic head | Open-source probabilistic | ✓ | Limited to univariate |
| BSQ | 2024/2025 | Vision tokenizer | Unit-sphere + sign() | 2.4× throughput, 100× compression | ✓ | No analysis on non-vision corpora |
| LFQ | 2024 | Vision tokenizer | Drop codebook + sign() | LLM beats diffusion at vis-gen | ✓ | No utilization diagnostics |
| FSQ | 2024 | Tokenizer | Per-dim rounding | ~100% codebook util | ✓ | Smaller effective vocab |
| VQ-VAE | 2017 | Tokenizer | Explicit codebook | Foundational | ✓ | Codebook collapse |
| Roaring Bitmaps | 2016 | Data systems | Compressed bitmaps | 2-100× over raw + bitmap | ✓ | Rarely in ML training |
| LSHBloom | 2024 | LLM dedup | MinHash + Bloom | Memory-efficient extreme-scale | ✓ | Dedup only, not training-loop |
| Dolma | 2024 | LLM corpus | Multi-stage dedup | 3T-token open corpus | ✓ | Pipeline, not method |
| S-OHEM | 2017 | Object detection | Stratified hard mining | Improves mAP | ✓ | Proposal-level, not token |
| "Beyond Random Sampling" | 2025 | LLM curriculum | Difficulty-ordered batches | 18-45% step reduction | partial | Document-level |
| "Early Quantization Shrinks Codebook" | 2026 | Tokenizer diag | Deferred quantization | Diagnostic suite + fix | — | Vision focus |

---

## 4. Where the User's Work Sits

| Existing method | Does our work do this? | Delta |
|-----------------|------------------------|-------|
| Kronos | Yes — we *use* Kronos as our target FM | We add diagnostic + stratified sampling infrastructure they didn't include |
| Chronos / Moirai / TimesFM | No | Orthogonal — they use different tokenization; our techniques port if they adopt BSQ-style discrete tokenization |
| BSQ | No (we don't modify BSQ) | We are the first to *measure* per-token utilization and hierarchical co-occurrence on financial BSQ data |
| FSQ / LFQ | No | Same orthogonal status — applicable if their corpus is sparse and clustered |
| VQ-VAE | No | Inapplicable: explicit codebook removes the bitmap sparsity benefit |
| Roaring Bitmaps | Yes — we *use* Roaring as our data structure | We apply Roaring to a non-standard ML target (BSQ posting lists) and propose two novel pipeline integrations |
| LSHBloom / Dolma | Partially | Their dedup is at document level; ours is at token-occurrence level. **Combinable.** |
| Lee et al. ACL 2022 (dedup) | Partial | Same idea (dedup helps), different scale and granularity. Cite as motivation. |
| S-OHEM | Partially | Same principle (stratified hard mining), different domain (token-level for time series, not proposal-level for detection). Cite as canonical prior art for stratified sampling. |
| "Beyond Random Sampling" | Partial | Same motivation (curriculum), different level (token vs document). Should cite. |
| Importance Sampling for Tail Risks | Partial | Finance-side motivation for sampling rare events more heavily. Useful framing reference. |
| "Early Quantization Shrinks Codebook" | Partial | Their gap call ("need diagnostic suites for discrete tokenizers") is *exactly* the gap our hierarchical-cooc diagnostic fills. **Cite explicitly as motivation.** |

**Verdict:** **Orthogonal to most existing methods, with deep complementarity to dedup + diagnostic literature.** The paper is not derivative — no existing work combines:
- Token-level (not document-level) stratified sampling
- Bitmap-backed (not hash-based) corpus indexing
- BSQ-specific (not general) diagnostic of hierarchical token utilization
- Financial-time-series target

This is a defensible "first work at the intersection" positioning. Reviewers can attack the *evidence* for the contribution (small backtest, single asset) but cannot easily attack *novelty*.

---

## 5. Gaps Worth Attacking (beyond what the paper already does)

### Gap A — Token-level deduplication at financial corpus scale
- **Statement:** No work has applied bitmap-backed near-duplicate detection at the token level within financial time-series corpora. Kronos pulls from 45 exchanges with massive overlapping symbols (BTC on Binance ≈ BTC on Coinbase ≈ BTC on Kraken).
- **Why it matters:** Dedup is a known win for text LLMs (Lee et al. 2022, Dolma). At 12B records and 45 exchanges, there is significant duplicate signal.
- **Why tractable:** MinHash + Roaring is straightforward to implement. Could leverage our existing posting-list code.
- **Reviewer-defensible:** Lee et al. 2022 shows >2% perplexity improvement from dedup; that's the precedent.

### Gap B — Cross-asset replication of hierarchical-diagnostic finding
- **Statement:** Our 52/163 deterministic-s2 finding is on BTC only. Replicating on equities (SPY), FX (EURUSD), commodities (Gold) would test whether the hierarchical-tightness phenomenon is universal across asset classes or BTC-specific.
- **Why tractable:** Tokenizer is frozen; only need data + a tokenization pass.

### Gap C — Vocabulary pruning based on cooc bitmaps
- **Statement:** If 32% of active s1 tokens have deterministic s2, the s2 head softmax over 1024 is wasted compute for those cases. Replace with a cached lookup.
- **Why it matters:** Could reduce inference FLOPs by ~10% on the autoregressive decode loop.
- **Why tractable:** Engineering work; no new training required.

### Gap D — Apply diagnostic across BSQ family on non-financial data
- **Statement:** Our hierarchical co-occurrence diagnostic is generic. Apply to image-BSQ (the original BSQ-ViT paper), video-BSQ (MagViT-v2 / LFQ), and speech codecs (RVQ-based). Compare hierarchical tightness across modalities.
- **Why it matters:** Could become a standard diagnostic in the discrete-tokenizer toolkit (cite "Early Quantization Shrinks Codebook" as the canonical call for such tools).

### Gap E — Multi-asset Qlib backtest replication
- **Statement:** Our backtest uses BTC only with a custom evaluation. Replicating Kronos's published Qlib backtest on Chinese A-share data would enable apples-to-apples comparison with their reported numbers.
- **Why tractable:** Their Qlib config is open-source; we already have the code in `kronos_src/`.

---

## 6. Open Questions / Disputes in the Field

- **BSQ vs FSQ vs LFQ — which discrete tokenization wins?** No clean comparison across all three on the same downstream task. Active debate in tokenization literature.
- **Curriculum learning for LLMs — does it survive scaling?** "Beyond Random Sampling" (2025) shows 18-45% step reduction *at small scale*. Whether this persists at trillion-token pretraining is contested.
- **Codebook collapse — design problem or training problem?** FSQ/LFQ argue it's a design problem (use codebook-free quantization). The collapse-mitigation literature (VQBridge, EdVAE) argues better training fixes it. Our position: *measure utilization on real downstream data, not just reconstruction.*
- **Data-loading bottlenecks — when do they matter?** Our A10 measurement says 0.1%. MegaScale (10k+ GPUs) and SSDTrain show it matters at extreme scale. The crossover point is unclear.

---

## 7. Recommended Reading Order

If a reader were to follow this paper's intellectual lineage:

1. **van den Oord 2017 (VQ-VAE)** — what tokenization originally meant
2. **Mentzer et al. 2024 (FSQ)** — why codebook-free quantization gained traction
3. **Yu et al. 2024 (MagViT-v2 / LFQ)** — what BSQ inherits architecturally
4. **Zhao et al. 2025 (BSQ)** — the actual quantization Kronos uses
5. **Shi et al. 2025 (Kronos)** — the foundation model we augment
6. **Lemire 2016 (Roaring Bitmaps)** — the data structure we apply
7. **Lee et al. 2022 (Dedup-helps-LLMs)** — corpus-side motivation for bitmap indexing
8. **arXiv 2603.17052 (Early Quantization Shrinks Codebook)** — the gap call for diagnostic tools
9. **Our paper** — bringing it all together

---

## 8. Bibliography (key 20 to actually cite in the paper)

```bibtex
@inproceedings{kronos2025,
  author    = {Shi, Yu and others},
  title     = {{Kronos}: A Foundation Model for the Language of Financial Markets},
  booktitle = {AAAI},
  year      = {2026},
  note      = {arXiv:2508.02739}
}

@inproceedings{bsq2024,
  author    = {Zhao, Yue and Xiong, Yuanjun and Kr{\"a}henb{\"u}hl, Philipp},
  title     = {Image and Video Tokenization with Binary Spherical Quantization},
  booktitle = {ICLR},
  year      = {2025},
  note      = {arXiv:2406.07548}
}

@article{roaring2016,
  author    = {Lemire, Daniel and Ssi-Yan-Kai, Gregory and Kaser, Owen},
  title     = {Consistently Faster and Smaller Compressed Bitmaps with {Roaring}},
  journal   = {Software: Practice and Experience},
  volume    = {46}, number = {11}, pages = {1547--1569},
  year      = {2016}
}

@inproceedings{chronos2024,
  author    = {Ansari, Abdul Fatir and Stella, Lorenzo and others},
  title     = {{Chronos}: Learning the Language of Time Series},
  booktitle = {arXiv:2403.07815},
  year      = {2024}
}

@inproceedings{moirai2024,
  author    = {Woo, Gerald and Liu, Chenghao and others},
  title     = {Unified Training of Universal Time Series Forecasting Transformers ({MOIRAI})},
  booktitle = {ICML},
  year      = {2024}
}

@inproceedings{timesfm2024,
  author    = {Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen},
  title     = {A Decoder-Only Foundation Model for Time-Series Forecasting},
  booktitle = {ICML},
  year      = {2024},
  note      = {arXiv:2310.10688}
}

@inproceedings{lagllama2023,
  author    = {Rasul, Kashif and Ashok, Arjun and others},
  title     = {Lag-{Llama}: Towards Foundation Models for Probabilistic Time Series Forecasting},
  booktitle = {arXiv:2310.08278},
  year      = {2023}
}

@inproceedings{bloomberggpt2023,
  author    = {Wu, Shijie and others},
  title     = {{BloombergGPT}: A Large Language Model for Finance},
  booktitle = {arXiv:2303.17564},
  year      = {2023}
}

@inproceedings{fingpt2023,
  author    = {Yang, Hongyang and Liu, Xiao-Yang and Wang, Christina Dan},
  title     = {{FinGPT}: Open-Source Financial Large Language Models},
  booktitle = {arXiv:2306.06031},
  year      = {2023}
}

@inproceedings{vqvae2017,
  author    = {van den Oord, A{\"a}ron and Vinyals, Oriol and Kavukcuoglu, Koray},
  title     = {Neural Discrete Representation Learning},
  booktitle = {NeurIPS},
  year      = {2017}
}

@inproceedings{fsq2024,
  author    = {Mentzer, Fabian and Minnen, David and Agustsson, Eirikur and Tschannen, Michael},
  title     = {Finite Scalar Quantization: {VQ-VAE} Made Simple},
  booktitle = {ICLR},
  year      = {2024},
  note      = {arXiv:2309.15505}
}

@inproceedings{lfq2024,
  author    = {Yu, Lijun and others},
  title     = {Language Model Beats Diffusion --- Tokenizer is Key to Visual Generation ({MagViT-v2}, {LFQ})},
  booktitle = {ICLR},
  year      = {2024},
  note      = {arXiv:2310.05737}
}

@inproceedings{lee2022dedup,
  author    = {Lee, Katherine and others},
  title     = {Deduplicating Training Data Makes Language Models Better},
  booktitle = {ACL},
  year      = {2022}
}

@article{lshbloom2024,
  author    = {Zelenfreund, Arham and others},
  title     = {{LSHBloom}: Memory-efficient, Extreme-scale Document Deduplication},
  journal   = {arXiv:2411.04257},
  year      = {2024}
}

@article{dolma2024,
  author    = {Soldaini, Luca and others},
  title     = {{Dolma}: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research},
  journal   = {arXiv:2402.00159},
  year      = {2024}
}

@inproceedings{sohem2017,
  author    = {Li, Zhaowei and Peng, Cheng and others},
  title     = {{S-OHEM}: Stratified Online Hard Example Mining for Object Detection},
  booktitle = {arXiv:1705.02233},
  year      = {2017}
}

@inproceedings{bengio2009curriculum,
  author    = {Bengio, Yoshua and others},
  title     = {Curriculum Learning},
  booktitle = {ICML},
  year      = {2009}
}

@article{curriculum_llm_2025,
  author    = {Anonymous},
  title     = {Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning},
  journal   = {arXiv:2506.11300},
  year      = {2025}
}

@article{shrinks_codebook_2026,
  author    = {Anonymous},
  title     = {Early Quantization Shrinks Codebook: A Simple Fix for Diversity-Preserving Tokenization},
  journal   = {arXiv:2603.17052},
  year      = {2026}
}

@article{megascale2024,
  author    = {Jiang, Ziheng and others},
  title     = {{MegaScale}: Scaling Large Language Model Training to More Than 10{,}000 {GPU}s},
  journal   = {arXiv:2402.15627},
  year      = {2024}
}
```

---

## Revision history

- 2026-05-24: Initial dossier, 20 references across 4 categories
