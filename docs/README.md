# Paper

`paper.tex` is the ICAIF~2026 submission in standard ACM SIGCONF format
(`acmart` document class with `sigconf` option). `paper.bib` is the
bibliography.

## Compile

You need a TeX distribution with `acmart`. On macOS:

```bash
brew install --cask mactex-no-gui   # or `mactex` for the full GUI
# Or, smaller:
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install acmart booktabs subcaption amsmath listings xcolor
```

Then compile:

```bash
cd docs
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Output: `paper.pdf`.

## Online compile

Easier: upload `paper.tex` + `paper.bib` to [Overleaf](https://overleaf.com).
The `acmart` template is built in — Overleaf will detect the document
class automatically.

## Figures referenced

The paper references these figures from the repo (currently not
embedded — uncomment the `\includegraphics` lines and copy the PNGs
into `docs/figures/`):

| Section | Figure | File |
|---------|--------|------|
| §4.2 | corpus token distribution | `experiments/token_distribution.png` |
| §4.5 | dataloader batch quality | `experiments/dataloader_benchmark.png` |
| §4.7 | A10 step-time breakdown | `cloud_benchmark/results/a10_kronos_base.png` |
| §4.8 | backtest comparison | `experiments/btc_backtest_comparison.png` |

Currently the paper uses tables only and references figure files only
in this README. To embed figures, add `\usepackage{graphicx}` (already
there) and lines like:

```latex
\begin{figure}[t]
  \includegraphics[width=\linewidth]{figures/token_distribution.png}
  \caption{...}
  \label{fig:tokens}
\end{figure}
```

## Submission checklist

Before submitting to ICAIF:

- [ ] Replace placeholder copyright with the conference-assigned one
- [ ] Add ORCID to author block
- [ ] Verify page count is within the conference limit (typically 8 + 1 for refs)
- [ ] Add CCS concepts under `\maketitle` if required
- [ ] Add the official ICAIF `\acmConference{}{}{}` line with year, month, location
- [ ] Embed figures (currently table-only)
- [ ] Run a spell check
- [ ] Re-run multi-seed experiments if a reviewer asks for them
