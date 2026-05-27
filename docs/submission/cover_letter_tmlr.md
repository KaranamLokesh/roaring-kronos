# TMLR Submission Cover Letter

**Title:** Measuring and Exploiting Token Sparsity in Financial Foundation Models: A Roaring-Bitmap Augmentation of Kronos

**Author:** Lokesh Karanam (Independent)
**Contact:** sunnylokesh3@gmail.com
**Submission date:** [DATE]

---

Dear TMLR Editors,

I am submitting the manuscript above for consideration. The paper applies the Roaring Bitmap compressed-integer-set data structure~[1] to the recently released Kronos financial foundation model~[2], which uses Binary Spherical Quantization (BSQ)~[3] tokenization, and reports three results spanning empirical measurement, training-pipeline infrastructure, and downstream evaluation.

## Why TMLR

TMLR's explicit emphasis on technical correctness over subjective significance is the right fit for this work. The paper contains:

- A clear empirical observation (extreme vocabulary sparsity and regime clustering in financial BSQ tokenization) that is novel for the financial-data setting but echoes known phenomena in vision tokenizers.
- A clean methodological contribution (token-level stratified sampling with bitmap-backed posting lists, plus a hierarchical co-occurrence diagnostic).
- Honest mixed results: the stratified sampler improves rare-token coverage and reduces drawdown in a fine-tune backtest, but does not show a clear aggregate-RankIC win on a single-asset, single-seed evaluation.
- An explicitly reported negative result (no wall-clock training speedup on an NVIDIA A10), with a careful discussion of the conditions under which the underlying CPU-side speedup would translate to wall-clock gains.

This combination — a real, measurable, reproducible contribution with honest limitations — is, in my reading of TMLR's mission, exactly the kind of paper TMLR exists to publish. Mainstream venues might either over-reject for "limited scope" or pressure the author to overclaim. TMLR's review philosophy accommodates the work as it stands.

## Specific contributions

1. **Empirical characterization.** First measurement of per-token utilization and hierarchical conditional sparsity for a BSQ-tokenized financial corpus. Of $2^{20} = 1{,}048{,}576$ possible tokens, only 417 (0.04%) appear in 17{,}352 BTC hourly bars; 52 of 163 active coarse tokens have a fully deterministic fine-token partner.

2. **Token-level stratified sampling.** A drop-in dataloader for Kronos's stage-2 fine-tuning that overweights rare-token windows via Roaring posting-list union. Improves per-batch rare-token coverage by 27% and reduces fine-tune backtest maximum drawdown by 2.7 percentage points versus uniform sampling, with zero measurable training-time overhead (measured on NVIDIA A10).

3. **Hierarchical co-occurrence diagnostic.** A general-purpose diagnostic for hierarchical BSQ-style tokenizers that directly answers the call of recent work [4] for diagnostic suites that go beyond aggregate reconstruction loss. To the best of my knowledge, this is the first such measurement on a non-vision corpus.

## Scope and limitations (transparent)

The paper is deliberately scoped to a single asset (BTC-USD hourly), single seed, single hardware tier (NVIDIA A10), and a single shock-fraction setting ($f=0.30$). These limitations are listed explicitly in Section 6. The contributions stand independently of the backtest outcome:

- The empirical characterization is a measurement, not a hypothesis test, and is not contingent on the fine-tune result.
- The infrastructure (posting lists, stratified sampler) is a tool, with quantified properties (11.4× lookup speedup, 1.86× compression, zero training overhead) that are independent of any downstream effect.
- The hierarchical diagnostic is a finding about the released Kronos tokenizer that does not depend on retraining.

I would welcome reviewer suggestions on which limitations are most important to address for revision.

## Code, data, reproducibility

All code, data (17{,}352 BTC hourly bars, tokenized and bitmapped), pre-trained checkpoints, and experiment scripts are released under MIT license at:

  https://github.com/KaranamLokesh/roaring-kronos

The repository contains complete reproduction scripts for every table and number in the paper, plus the cloud-benchmark script used to produce the A10 timing measurements.

## Conflicts and prior submissions

This work has not been submitted, in whole or in part, to any other venue. It will be made available as an arXiv preprint concurrent with TMLR submission. No conflicts of interest to declare.

## Suggested action editors / reviewers

I do not have specific reviewer suggestions, but reviewers with expertise in any of the following would be appropriate:
- Discrete tokenization for non-text modalities (BSQ, FSQ, LFQ, RVQ)
- Time-series foundation models (Kronos, Chronos, Moirai, TimesFM, Lag-Llama)
- Quantitative finance applied to deep learning
- Compressed data structures and indexing systems (Roaring, MinHash, Bloom)

Thank you for considering the manuscript.

Sincerely,

Lokesh Karanam
sunnylokesh3@gmail.com

---

### References cited in this letter
[1] Lemire et al. *Consistently Faster and Smaller Compressed Bitmaps with Roaring*. Software: Practice and Experience, 2016.
[2] Shi et al. *Kronos: A Foundation Model for the Language of Financial Markets*. AAAI 2026.
[3] Zhao, Xiong, Krähenbühl. *Image and Video Tokenization with Binary Spherical Quantization*. ICLR 2025.
[4] *Early Quantization Shrinks Codebook: A Simple Fix for Diversity-Preserving Tokenization*. arXiv 2603.17052.
