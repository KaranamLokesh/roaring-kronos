#!/bin/bash
#
# Build the arXiv submission tarball from docs/paper.tex.
#
# Usage (from repo root):
#   bash docs/submission/prep_arxiv.sh
#
# Output:
#   docs/submission/arxiv_submission.tar.gz   — upload this file to arxiv.org/submit
#   docs/submission/paper.pdf                  — local preview
#
# Requires: pdflatex, bibtex (any standard TeX Live or MacTeX install).

set -e

# Resolve repo root regardless of where script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
SUBMIT_DIR="$REPO_ROOT/docs/submission"

cd "$DOCS_DIR"

echo "════════════════════════════════════════════════════════"
echo "  arXiv submission package builder"
echo "════════════════════════════════════════════════════════"
echo ""

# Sanity check
if [ ! -f paper.tex ]; then
    echo "ERROR: paper.tex not found in $DOCS_DIR"
    exit 1
fi
if [ ! -f paper.bib ]; then
    echo "ERROR: paper.bib not found in $DOCS_DIR"
    exit 1
fi

# Check for pdflatex
if ! command -v pdflatex >/dev/null 2>&1; then
    echo "WARNING: pdflatex not found on PATH."
    echo "Install MacTeX (brew install --cask mactex) or BasicTeX,"
    echo "or use Overleaf to compile and download the PDF manually."
    echo ""
    echo "Skipping local build — will still package source files."
    SKIP_BUILD=1
else
    SKIP_BUILD=0
fi

# Stage 1: local PDF preview build
if [ "$SKIP_BUILD" = "0" ]; then
    echo "[1/3] Building local PDF preview…"
    rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf

    pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 || true
    bibtex   paper                              >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1 || true

    if [ ! -f paper.pdf ]; then
        echo "  ✗ PDF build failed. Check paper.log for errors."
        echo "  You can still upload the source tarball; arXiv will rebuild."
    else
        cp paper.pdf "$SUBMIT_DIR/paper.pdf"
        PAGES=$(pdfinfo paper.pdf 2>/dev/null | awk '/^Pages:/ {print $2}')
        echo "  ✓ PDF built: $SUBMIT_DIR/paper.pdf  (${PAGES:-?} pages)"
    fi
fi

# Stage 2: build the source tarball arXiv expects
echo ""
echo "[2/3] Packaging source files for arXiv…"

STAGING="$SUBMIT_DIR/_staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"

# Copy only the files arXiv needs (NOT the build artifacts)
cp paper.tex paper.bib "$STAGING/"

# Copy figure files (referenced via \includegraphics{figures/...})
if [ -d figures ]; then
    mkdir -p "$STAGING/figures"
    cp figures/*.png "$STAGING/figures/" 2>/dev/null || true
    N_FIGS=$(ls "$STAGING/figures/" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ Included $N_FIGS figure(s) from figures/"
fi

# Include the .bbl too — arXiv may not run BibTeX itself for older
# submissions, and including the compiled .bbl avoids that risk.
if [ -f paper.bbl ]; then
    cp paper.bbl "$STAGING/"
    echo "  ✓ Included paper.bbl (compiled bibliography)"
fi

# Build tarball
cd "$STAGING"
tar -czf "$SUBMIT_DIR/arxiv_submission.tar.gz" * 2>/dev/null

cd "$DOCS_DIR"
rm -rf "$STAGING"

SIZE=$(du -h "$SUBMIT_DIR/arxiv_submission.tar.gz" | cut -f1)
echo "  ✓ Tarball ready: $SUBMIT_DIR/arxiv_submission.tar.gz  ($SIZE)"

# Stage 3: cleanup local build artifacts in docs/
echo ""
echo "[3/3] Cleaning build artifacts…"
rm -f paper.aux paper.bbl paper.blg paper.log paper.out
echo "  ✓ Cleaned aux/log/etc."

echo ""
echo "════════════════════════════════════════════════════════"
echo "Done. Next steps:"
echo ""
echo "  1. Preview the PDF:"
echo "       open '$SUBMIT_DIR/paper.pdf'"
echo ""
echo "  2. Submit to arXiv:"
echo "       open https://arxiv.org/submit"
echo "       Upload: $SUBMIT_DIR/arxiv_submission.tar.gz"
echo "       Use metadata from: $SUBMIT_DIR/arxiv_metadata.md"
echo ""
echo "  3. After arXiv ID is assigned, submit to TMLR via OpenReview:"
echo "       https://openreview.net/group?id=TMLR"
echo "       Upload: $SUBMIT_DIR/paper.pdf"
echo "       Cover letter from: $SUBMIT_DIR/cover_letter_tmlr.md"
echo "════════════════════════════════════════════════════════"
