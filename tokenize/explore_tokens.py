"""
Download BTC-USD hourly data, tokenize with Kronos BSQ tokenizer, and analyze
the resulting token distribution. This is our first look at what the tokens
actually look like before building any Roaring Bitmap infrastructure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kronos_src'))

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import KronosTokenizer


# ── Config ────────────────────────────────────────────────────────────────────
TICKER      = "BTC-USD"
INTERVAL    = "1h"
PERIOD      = "2y"          # yfinance max for 1h is 730 days
CLIP        = 5.0
CHUNK_SIZE  = 512           # match Kronos max context
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
PLOT_PATH   = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'token_distribution.png')


# ── 1. Fetch data ─────────────────────────────────────────────────────────────
print(f"[1/5] Fetching {TICKER} {INTERVAL} data ({PERIOD})…")
raw = yf.download(TICKER, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
raw = raw.dropna()
# yfinance may return MultiIndex columns like ('Close', 'BTC-USD') — flatten to just the field name
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = [c[0].lower() for c in raw.columns]
else:
    raw.columns = [c.lower() for c in raw.columns]

# yfinance gives Open/High/Low/Close/Volume — we derive Amount = Volume * avg_price
raw['amount'] = raw['volume'] * (raw[['open','high','low','close']].mean(axis=1))

df = raw[['open','high','low','close','volume','amount']].copy()
df.index.name = 'timestamp'
df = df.reset_index()

print(f"    {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(os.path.join(OUTPUT_DIR, 'btc_1h.csv'), index=False)
print(f"    Saved to data/btc_1h.csv")


# ── 2. Preprocess exactly as KronosPredictor does ─────────────────────────────
print("\n[2/5] Preprocessing (z-score normalize, clip to ±5)…")
price_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
x = df[price_cols].values.astype(np.float32)

x_mean = np.mean(x, axis=0)
x_std  = np.std(x,  axis=0)
x_norm = (x - x_mean) / (x_std + 1e-5)
x_norm = np.clip(x_norm, -CLIP, CLIP)

print(f"    Shape: {x_norm.shape}  |  Columns: {price_cols}")
print(f"    Normalized close — mean={x_norm[:,3].mean():.3f}, std={x_norm[:,3].std():.3f}, "
      f"min={x_norm[:,3].min():.3f}, max={x_norm[:,3].max():.3f}")


# ── 3. Load tokenizer ─────────────────────────────────────────────────────────
print("\n[3/5] Loading KronosTokenizer (NeoQuasar/Kronos-Tokenizer-base)…")
device = "mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu"
print(f"    Device: {device}")

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
tokenizer = tokenizer.to(device)
tokenizer.eval()

s1_bits = tokenizer.s1_bits
s2_bits = tokenizer.s2_bits
vocab_s1 = 2 ** s1_bits    # 32
vocab_s2 = 2 ** s2_bits    # 16
vocab_full = vocab_s1 * vocab_s2  # 512

print(f"    s1_bits={s1_bits}, s2_bits={s2_bits}  →  vocab_s1={vocab_s1}, vocab_s2={vocab_s2}, full={vocab_full}")


# ── 4. Encode in sliding windows ──────────────────────────────────────────────
print(f"\n[4/5] Encoding {len(x_norm)} bars in chunks of {CHUNK_SIZE}…")

all_s1, all_s2 = [], []

with torch.no_grad():
    # Encode the whole sequence in one shot (fits in memory for ~17k bars)
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)  # (1, T, 6)
    indices = tokenizer.encode(x_tensor, half=True)               # [s1(1,T), s2(1,T)]
    s1_tokens = indices[0].squeeze(0).cpu().numpy()               # (T,)
    s2_tokens = indices[1].squeeze(0).cpu().numpy()               # (T,)

full_tokens = s1_tokens * vocab_s2 + s2_tokens   # combined 9-bit index in [0, 511]

print(f"    Encoded {len(full_tokens)} tokens")
print(f"    s1 range: [{s1_tokens.min()}, {s1_tokens.max()}]  (expected [0, {vocab_s1-1}])")
print(f"    s2 range: [{s2_tokens.min()}, {s2_tokens.max()}]  (expected [0, {vocab_s2-1}])")
print(f"    full token range: [{full_tokens.min()}, {full_tokens.max()}]  (expected [0, {vocab_full-1}])")

# Save raw tokens
np.save(os.path.join(OUTPUT_DIR, 'btc_1h_s1_tokens.npy'), s1_tokens)
np.save(os.path.join(OUTPUT_DIR, 'btc_1h_s2_tokens.npy'), s2_tokens)
np.save(os.path.join(OUTPUT_DIR, 'btc_1h_full_tokens.npy'), full_tokens)
print(f"    Tokens saved to data/")


# ── 5. Analyze distributions ──────────────────────────────────────────────────
print("\n[5/5] Analyzing token distributions…")

# Frequency counts
s1_counts  = Counter(s1_tokens.tolist())
s2_counts  = Counter(s2_tokens.tolist())
full_counts = Counter(full_tokens.tolist())

s1_freq  = np.array([s1_counts.get(i, 0)  for i in range(vocab_s1)], dtype=np.float64)
s2_freq  = np.array([s2_counts.get(i, 0)  for i in range(vocab_s2)], dtype=np.float64)
full_freq = np.array([full_counts.get(i, 0) for i in range(vocab_full)], dtype=np.float64)

total = len(full_tokens)

print(f"\n  ── s1 (coarse, {vocab_s1} tokens) ──")
print(f"    Unique tokens used:  {len(s1_counts)} / {vocab_s1}")
print(f"    Top 5 by frequency:")
for tok, cnt in s1_counts.most_common(5):
    print(f"      s1={tok:2d}  count={cnt:5d}  ({100*cnt/total:.1f}%)")
print(f"    Entropy: {-np.sum((s1_freq/total)*np.log2(s1_freq/total + 1e-10)):.3f} bits  (max={np.log2(vocab_s1):.3f})")

print(f"\n  ── s2 (fine, {vocab_s2} tokens) ──")
print(f"    Unique tokens used:  {len(s2_counts)} / {vocab_s2}")
print(f"    Top 5 by frequency:")
for tok, cnt in s2_counts.most_common(5):
    print(f"      s2={tok:2d}  count={cnt:5d}  ({100*cnt/total:.1f}%)")
print(f"    Entropy: {-np.sum((s2_freq/total)*np.log2(s2_freq/total + 1e-10)):.3f} bits  (max={np.log2(vocab_s2):.3f})")

print(f"\n  ── full 9-bit token ({vocab_full} tokens) ──")
print(f"    Unique tokens used:  {len(full_counts)} / {vocab_full}")
top5_full = full_counts.most_common(5)
print(f"    Top 5 by frequency:")
for tok, cnt in top5_full:
    print(f"      token={tok:3d}  (s1={tok//vocab_s2}, s2={tok%vocab_s2})  count={cnt:5d}  ({100*cnt/total:.1f}%)")
bottom5_full = full_counts.most_common()[-5:]
print(f"    Bottom 5 (rarest):")
for tok, cnt in bottom5_full:
    print(f"      token={tok:3d}  (s1={tok//vocab_s2}, s2={tok%vocab_s2})  count={cnt:4d}  ({100*cnt/total:.2f}%)")
print(f"    Full entropy: {-np.sum((full_freq/total)*np.log2(full_freq/total + 1e-10)):.3f} bits  (max={np.log2(vocab_full):.3f})")

# Vocabulary coverage
never_seen = vocab_full - len(full_counts)
print(f"\n    Never-seen tokens:   {never_seen} / {vocab_full}  ({100*never_seen/vocab_full:.1f}% of vocab unused)")

# Run-length stats (regime duration)
# How many consecutive bars share the same s1 (coarse mood)?
runs = []
cur = s1_tokens[0]; run_len = 1
for t in s1_tokens[1:]:
    if t == cur:
        run_len += 1
    else:
        runs.append(run_len)
        cur = t; run_len = 1
runs.append(run_len)
runs = np.array(runs)
print(f"\n  ── s1 run-length stats (consecutive same coarse mood) ──")
print(f"    Total runs:  {len(runs)}")
print(f"    Mean length: {runs.mean():.2f} bars")
print(f"    Median:      {np.median(runs):.0f} bars")
print(f"    Max:         {runs.max()} bars")
pct_run_3plus = 100 * (runs >= 3).sum() / len(runs)
print(f"    Runs ≥ 3 bars: {pct_run_3plus:.1f}%  ← Roaring Run Container territory")

# Co-occurrence: for each s1, which s2 values appear?
print(f"\n  ── Hierarchical co-occurrence (s2 vocab per s1 context) ──")
cooccurrence = {}
for s1v in range(vocab_s1):
    mask = (s1_tokens == s1v)
    s2_vals = set(s2_tokens[mask].tolist()) if mask.any() else set()
    cooccurrence[s1v] = s2_vals

coverages = [len(v) / vocab_s2 for v in cooccurrence.values()]
active_s1 = [s1v for s1v in range(vocab_s1) if cooccurrence[s1v]]
print(f"    Active s1 values: {len(active_s1)} / {vocab_s1}")
print(f"    Mean s2 coverage per s1: {np.mean(coverages)*100:.1f}% of {vocab_s2} fine tokens")
print(f"    Min s2 coverage: {min(coverages)*100:.1f}%  Max: {max(coverages)*100:.1f}%")
tightest = min(active_s1, key=lambda v: len(cooccurrence[v]))
loosest  = max(active_s1, key=lambda v: len(cooccurrence[v]))
print(f"    Tightest s1={tightest}: only {len(cooccurrence[tightest])} fine tokens  → {sorted(cooccurrence[tightest])}")
print(f"    Loosest  s1={loosest}:  {len(cooccurrence[loosest])} fine tokens")


# ── 6. Plot ───────────────────────────────────────────────────────────────────
print(f"\nGenerating plots → {PLOT_PATH}")
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

# s1 distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(range(vocab_s1), s1_freq / total * 100, color='steelblue', width=0.8)
ax1.set_title(f's1 (coarse) token distribution — {vocab_s1} tokens', fontsize=11)
ax1.set_xlabel('s1 token id'); ax1.set_ylabel('% of bars'); ax1.grid(axis='y', alpha=0.4)

# s2 distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(range(vocab_s2), s2_freq / total * 100, color='coral', width=0.8)
ax2.set_title(f's2 (fine) token distribution — {vocab_s2} tokens', fontsize=11)
ax2.set_xlabel('s2 token id'); ax2.set_ylabel('% of bars'); ax2.grid(axis='y', alpha=0.4)

# Full token distribution (sorted by frequency)
ax3 = fig.add_subplot(gs[1, :])
sorted_full = np.sort(full_freq)[::-1]
ax3.bar(range(len(sorted_full)), sorted_full / total * 100, color='mediumseagreen', width=1.0)
ax3.set_title(f'Full 9-bit token distribution (sorted by frequency) — {len(full_counts)} of {vocab_full} tokens seen', fontsize=11)
ax3.set_xlabel('rank'); ax3.set_ylabel('% of bars'); ax3.grid(axis='y', alpha=0.4)
# Mark the 80th percentile token
cumsum = np.cumsum(sorted_full / total)
p80_idx = np.searchsorted(cumsum, 0.80)
ax3.axvline(p80_idx, color='red', linestyle='--', alpha=0.7, label=f'Top-{p80_idx+1} tokens = 80% of data')
ax3.legend(fontsize=9)

# Run-length distribution (capped at 20)
ax4 = fig.add_subplot(gs[2, 0])
run_cap = min(runs.max(), 30)
run_hist = np.bincount(np.minimum(runs, run_cap), minlength=run_cap+1)[1:]
ax4.bar(range(1, run_cap+1), run_hist, color='orchid', width=0.8)
ax4.set_title('s1 run-length distribution (consecutive same coarse mood)', fontsize=11)
ax4.set_xlabel('run length (bars)'); ax4.set_ylabel('count'); ax4.grid(axis='y', alpha=0.4)

# s2 vocab coverage per s1
ax5 = fig.add_subplot(gs[2, 1])
cov_counts = [len(cooccurrence[s1v]) for s1v in range(vocab_s1)]
ax5.bar(range(vocab_s1), cov_counts, color='goldenrod', width=0.8)
ax5.axhline(vocab_s2, color='red', linestyle='--', alpha=0.6, label=f'Full s2 vocab ({vocab_s2})')
ax5.set_title('Unique s2 tokens seen per s1 context', fontsize=11)
ax5.set_xlabel('s1 token id'); ax5.set_ylabel('# distinct s2 values'); ax5.grid(axis='y', alpha=0.4)
ax5.legend(fontsize=9)

fig.suptitle(f'Kronos BSQ Token Analysis — BTC-USD 1h ({total} bars)', fontsize=13, fontweight='bold')
plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight')
print(f"Saved plot to {PLOT_PATH}")

print("\nDone. Key takeaways:")
print(f"  • {len(full_counts)}/{vocab_full} full tokens ever seen on BTC 1h")
pct_top10 = 100 * sum(v for _, v in full_counts.most_common(10)) / total
print(f"  • Top 10 tokens account for {pct_top10:.1f}% of all bars  ← heavy concentration")
print(f"  • p80 token rank: {p80_idx+1}  (80% of bars covered by only {p80_idx+1} token types)")
print(f"  • Mean s1 run length: {runs.mean():.1f} bars  ← regime clustering = Roaring Run Container payoff")
