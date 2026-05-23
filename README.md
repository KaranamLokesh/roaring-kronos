# Roaring Kronos

Applying Roaring Bitmaps to optimize the Kronos financial time-series foundation model.

## Overview

[Kronos](https://github.com/shiyu-coder/Kronos) is a decoder-only transformer for financial candlestick data that uses **Binary Spherical Quantization (BSQ)** to tokenize OHLCVA bars into 9-bit tokens (512 possible "market moods").

This project integrates Roaring Bitmap techniques into Kronos's training pipeline to improve efficiency without sacrificing model quality.

## Key Ideas

### Where Roaring Bitmaps Help

1. **Rare-event sampling** — Build one Roaring bitmap per token (out of 512) over the full training corpus. Instantly retrieve all "crash bars" or "spike bars" for oversampling during training, without scanning 12B records.

2. **Deduplication** — Kronos trains on 45+ exchanges with overlapping symbols. Bitmap-based dedup removes near-duplicate bars before training.

3. **Attention mask compression** — Compress causal attention masks in the stage-2 predictor transformer (applies directly from prior work).

4. **Gradient sparsification** — Top-K sparse gradients during DDP training, compressed with Roaring bitmaps.

5. **Hierarchical diagnostic** — Track which fine tokens (s2, 4 bits) follow each coarse token (s1, 5 bits) via per-coarse Roaring bitmaps. Measures how "tight" the learned BSQ hierarchy actually is.

## BSQ Quick Reference

- Each candlestick → 9 yes/no bits → 1 of 512 tokens
- Hierarchical: 5 coarse bits (s1) → 32 moods, then 4 fine bits (s2) → 16 refinements
- Pre-trained checkpoints: `NeoQuasar/Kronos-{mini,small,base,large}` on Hugging Face

## Project Structure

```
roaring-kronos/
├── data/               # Data fetching and preprocessing
├── tokenize/           # BSQ tokenization pipeline
├── bitmaps/            # Roaring bitmap construction and analysis
├── training/           # Modified Kronos training with Roaring optimizations
├── eval/               # Qlib-based backtest evaluation
├── experiments/        # Experiment configs and results
└── notebooks/          # Exploration notebooks
```

## Planned Experiments

| Experiment | Metric | Baseline |
|---|---|---|
| Token corpus bitmap compression | Compression ratio | Raw uint16 list |
| Rare-event oversampling | Training loss on shock bars | Uniform sampling |
| Deduplication at scale | % duplicate bars removed | No dedup |
| Attention mask compression | Mask size, step time | Dense mask |
| Gradient sparsification | Gradient size, throughput | Dense gradients |
| Hierarchical diagnostic | Run/Array container ratio per s1 | N/A |
| End-to-end quality | Qlib IC, cumulative return | Vanilla Kronos |

## Setup

```bash
pip install -r requirements.txt
```

## References

- [Kronos paper (AAAI 2026)](https://arxiv.org/abs/...) — shiyu-coder/Kronos
- [BSQ paper](https://arxiv.org/abs/2406.07548) — Binary Spherical Quantization
- [PyRoaringBitMap](https://github.com/Ezibenroc/PyRoaringBitMap)
