# arXiv Submission Metadata

Ready-to-paste fields for the arXiv submission form at https://arxiv.org/submit

---

## Title

```
Measuring and Exploiting Token Sparsity in Financial Foundation Models: A Roaring-Bitmap Augmentation of Kronos
```

---

## Authors

```
Lokesh Karanam
```

For the affiliation field: leave blank or use `Independent Researcher`.

---

## Abstract (copy verbatim)

```
Financial foundation models such as Kronos adopt Binary Spherical Quantization (BSQ) to convert continuous OHLCVA candlestick bars into discrete tokens drawn from a vocabulary of size 2^K, treating market prediction as language modeling on a learned alphabet. While the BSQ design literature focuses on codebook utilization on image and video reconstruction, the structure of the financial corpus produced by such tokenizers has not been empirically characterized. We report the first such measurement. On 17,352 BTC-USD hourly bars over two years, the trained Kronos BSQ tokenizer uses only 417 of 1,048,576 possible tokens (0.04%); the top 49 tokens cover 80% of all bars; and the mean s1 run-length is 2.4 bars with a maximum of 80 bars. The corpus is exactly the regime-clustered, sparse integer set for which Roaring Bitmaps -- the standard data structure for compressed integer indexes in production search and analytics systems -- were designed.

We contribute three results. (i) A token-level stratified shock-bar dataloader, built atop Roaring posting lists, increases per-batch rare-token coverage by 27% with zero measurable training overhead on an NVIDIA A10. (ii) A novel hierarchical co-occurrence diagnostic for BSQ tokenizers reveals that 52 of 163 active coarse tokens are paired with a deterministic fine token -- the BSQ hierarchy's conditional sparsity is far tighter than any prior measurement on a non-vision corpus suggests. (iii) In a fine-tune backtest matching Kronos's published evaluation protocol, our stratified sampler achieves the best single-month RankIC of any variant (April 2026, +0.195) and reduces maximum drawdown by 2.7 percentage points versus uniform sampling, with no aggregate signal-quality penalty. All code, data, and bitmaps are released under MIT license.
```

---

## Primary subject category

```
cs.LG  (Machine Learning)
```

## Cross-list categories

In order of relevance:

```
q-fin.ST  (Statistical Finance)
cs.IR     (Information Retrieval) -- for the bitmap/indexing angle
cs.DS     (Data Structures and Algorithms) -- for Roaring
q-fin.CP  (Computational Finance)
```

When the form asks, list `cs.LG` as primary and the others as additional/cross-list categories. The first two (`cs.LG` + `q-fin.ST`) are essential. The others are optional but help discoverability.

---

## Comments

This field appears below the abstract. Use it to declare submission status:

```
Submitted to TMLR. 12 pages main + references. Code, data, and bitmaps released at https://github.com/KaranamLokesh/roaring-kronos under MIT license.
```

---

## License

Choose `arXiv.org perpetual non-exclusive license` (the default, recommended). This lets you submit to any journal afterward.

---

## MSC / ACM classification (optional)

If asked:

- **ACM CCS:** Computing methodologies → Machine learning → Learning paradigms → Supervised learning
- **MSC:** 68T07 (Artificial neural networks and deep learning) plus 91G70 (Statistical methods, econometrics for finance)

Skip these if the form makes them optional — they're nice-to-have, not required.

---

## Files to upload

The arXiv accepts the LaTeX source (preferred) or a single PDF.

**Option A — LaTeX source (preferred, allows arXiv to rebuild the PDF correctly):**

Upload a single tar.gz containing:
```
paper.tex
paper.bib
ACM-Reference-Format.bst   (if not already in your TeX install)
acmart.cls                  (only if you've customised it; otherwise arXiv has it)
```

Use the build script in this directory: `bash docs/submission/prep_arxiv.sh`

**Option B — Single PDF:**

If LaTeX compilation on arXiv fails, fall back to uploading `paper.pdf` directly.

---

## After submission

You'll get an arXiv ID like `2606.XXXXX`. Update these places:

1. **Repo README.md** — add a "Cite as" section with the arXiv link
2. **TMLR submission** — note the arXiv ID in your cover letter
3. **Your Google Scholar profile** — claim the paper once it appears in the index

---

## Quick checklist

- [ ] Build PDF locally (or in Overleaf) — verify it compiles cleanly
- [ ] Title field copied
- [ ] Author field copied
- [ ] Abstract copied (TextEdit, not Word, to avoid smart quotes)
- [ ] Subject categories selected (`cs.LG` primary + `q-fin.ST` cross-list)
- [ ] Comments field filled
- [ ] Source files tarball ready
- [ ] License: arXiv perpetual non-exclusive (default)
- [ ] Submit
- [ ] Wait 1-24 hours for the moderation pass (first-time submitters take longer)
