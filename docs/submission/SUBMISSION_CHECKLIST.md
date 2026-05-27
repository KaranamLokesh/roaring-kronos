# Submission Checklist

A step-by-step from "paper is written" to "submitted to TMLR with arXiv preprint."

## Phase 1 — Pre-submission polish (1-2 hours)

- [ ] Re-read `docs/paper.tex` end-to-end one final time
- [ ] Spell-check (open in any editor with spellcheck enabled)
- [ ] Verify all numerical claims match the actual experimental outputs in `experiments/`
- [ ] Verify all references in `paper.bib` are correct (especially arXiv IDs — they're easy to typo)
- [ ] Embed figures? (Currently table-only. Consider adding `experiments/token_distribution.png` as Figure 1 and `experiments/btc_backtest_comparison.png` as Figure 2 — see notes below.)
- [ ] Compile the PDF and visually scan every page for layout issues

### Figure embedding (optional but recommended)

If you want to add figures, add these blocks at the right places in `paper.tex`:

```latex
% Goes in §4.2 Corpus characterization
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/token_distribution.png}
  \caption{Token distribution on 17,352 BTC hourly bars.
  Top-left: $s_1$ frequency. Top-right: $s_2$ frequency.
  Middle: full-token frequency rank curve.
  Bottom-left: $s_1$ run-length distribution.
  Bottom-right: $s_2$ vocab usage per $s_1$.}
  \label{fig:tokens}
\end{figure}
```

```latex
% Goes in §4.8 Fine-tune backtest
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/btc_backtest_comparison.png}
  \caption{Three-way fine-tune backtest comparison on BTC-USD
  hourly test split. Roaring-FT records the best single-month
  RankIC of any variant (April 2026, $+0.195$).}
  \label{fig:backtest}
\end{figure}
```

Then copy the PNGs into `docs/figures/`:
```bash
mkdir -p docs/figures
cp experiments/token_distribution.png docs/figures/
cp experiments/btc_backtest_comparison.png docs/figures/
```

## Phase 2 — Build the arXiv tarball (15 min)

### Option A — Local build (requires MacTeX or BasicTeX)

```bash
cd /Users/lokeshkaranam/Desktop/Misc-projects/roaring-kronos
bash docs/submission/prep_arxiv.sh
```

This produces:
- `docs/submission/paper.pdf` — preview
- `docs/submission/arxiv_submission.tar.gz` — upload this

### Option B — Overleaf build (no TeX install needed)

1. Go to https://overleaf.com and create a new project
2. Upload `docs/paper.tex` and `docs/paper.bib` (drag-drop into the file panel)
3. Click "Recompile" — the document class is `acmart`, which Overleaf has built-in
4. Verify the PDF compiles cleanly
5. Click Menu → "Source" → Download as zip → rename to `arxiv_submission.tar.gz`

## Phase 3 — Submit to arXiv (30 min including queue time)

1. Create an arXiv account if you don't have one: https://arxiv.org/user/register
   - **First-time submitters need endorsement** for `cs.LG` and `q-fin.ST` categories. If you don't have an endorser, the submission will be held until arXiv staff verify your identity (usually 1-3 days). Plan for this delay.

2. Go to https://arxiv.org/submit and start a new submission.

3. Use the metadata from `docs/submission/arxiv_metadata.md`:
   - Copy the title (one line)
   - Copy the author name (`Lokesh Karanam`)
   - Copy the abstract verbatim into the abstract field
   - Set primary category to `cs.LG`
   - Add cross-list categories `q-fin.ST`, `cs.IR`, `cs.DS`
   - Copy the "Comments" text into the comments field
   - License: leave at default (arXiv perpetual non-exclusive)

4. Upload the tarball or the PDF.

5. Click "Submit" and wait for the arXiv ID to appear (usually within 24 hours, may take longer for first submissions pending endorsement).

6. Once you have your arXiv ID (format: `26MM.NNNNN`), note it for use in TMLR.

## Phase 4 — Submit to TMLR (45 min)

1. Create an OpenReview account if you don't have one: https://openreview.net/signup

2. Go to https://openreview.net/group?id=TMLR

3. Click "Submit New Manuscript" (the button is visible to logged-in users).

4. Fill the OpenReview form:
   - **Title:** copy from `arxiv_metadata.md`
   - **Authors:** Lokesh Karanam — link your OpenReview profile
   - **Abstract:** copy from `arxiv_metadata.md`
   - **Keywords:** financial foundation models, time-series tokenization, BSQ, Roaring Bitmaps, stratified sampling, codebook utilization, Kronos
   - **TL;DR (one sentence):** "We measure extreme vocabulary sparsity and hierarchical conditional sparsity in the Kronos financial foundation model's BSQ-tokenized corpus, and use Roaring Bitmaps to build a zero-overhead stratified sampler that reduces fine-tune drawdown by 2.7 percentage points."
   - **PDF upload:** `docs/submission/paper.pdf`
   - **Supplementary material:** optional, can attach the full reproduction zip

5. **Cover letter field (or separate file upload depending on the OpenReview form):**
   Copy/paste the entire content of `docs/submission/cover_letter_tmlr.md`. If TMLR's form has a separate "Comments to Action Editor" field, paste the cover letter there. Update `[DATE]` to today's date and add the arXiv ID once assigned (e.g., "Also available as arXiv:26MM.NNNNN").

6. **Code link field:** `https://github.com/KaranamLokesh/roaring-kronos`

7. Verify all fields, submit.

8. You'll get a submission ID; reviews arrive within ~4 weeks.

## Phase 5 — After submission

- [ ] Add the arXiv link to the repo README ("Cite as: arXiv:26MM.NNNNN")
- [ ] Tweet/post about it if you want to (optional)
- [ ] Start the multi-seed and shock_frac sweep experiments in parallel — these are the most likely reviewer-requested revisions and being ready saves weeks during revision

## Common pitfalls (avoid these)

- **Do not** submit to TMLR before the arXiv version is live. TMLR uses double-blind review; if your paper is on arXiv first, reviewers may find it. That's fine — TMLR has stated this does not affect review outcomes — but make sure the arXiv version matches the TMLR submission exactly.
- **Do not** put your real name in the file names if you anonymize. TMLR may strip your name from the submission but include it in metadata.
- **Do not** include figures that aren't yours without proper attribution. Figures from Kronos / BSQ papers should be clearly cited; ideally use only figures you generated yourself.
- **Do not** click submit on arXiv twice — duplicate submissions get flagged and the second one is rejected.
- **Do** save the OpenReview submission ID immediately. If you lose it, contact is harder.

## Estimated total time

- Phase 1 (polish): 1-2 hours
- Phase 2 (build): 15 min
- Phase 3 (arXiv): 30 min active + 1-72 hours queue
- Phase 4 (TMLR): 45 min
- Phase 5: ongoing

You can do all of Phases 1-4 in a single ~3 hour evening session, assuming arXiv endorsement is not blocking.
