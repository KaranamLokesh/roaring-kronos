"""
Benchmark: VanillaDataLoader vs RoaringDataLoader

Measures three things:
  1. Throughput (batches/sec, samples/sec)
  2. Rare-token rate in batches  — how often a shock bar appears per window
  3. Token distribution entropy  — how much variety the model actually sees

Run from repo root:
  PYENV_VERSION=3.10.14 python training/benchmark_dataloader.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from training.roaring_dataloader import make_vanilla_loader, make_roaring_loader

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN          = 128
BATCH_SIZE       = 32
STEPS_PER_EPOCH  = 256      # batches to consume per run
SHOCK_FRAC       = 0.3      # 30 % of roaring windows anchored at shock bars
RARE_THRESHOLD   = 5        # tokens appearing ≤ 5 times = "rare"
PLOT_PATH        = os.path.join(os.path.dirname(__file__), '..', 'experiments',
                                'dataloader_benchmark.png')

# pre-load full token array once (shared)
import numpy as _np
ft_all = _np.load(os.path.join(os.path.dirname(__file__), '..', 'data',
                                'btc_1h_full_tokens.npy'))
ft_counts_global = Counter(ft_all.tolist())
rare_set_global  = {tok for tok, cnt in ft_counts_global.items() if cnt <= RARE_THRESHOLD}
print(f"Rare tokens: {len(rare_set_global)} | Total corpus: {len(ft_all):,} bars\n")


def run_loader(loader, label: str):
    """Consume all batches, measure throughput and token stats."""
    all_full_tokens = []
    rare_hits_per_window = []

    t0 = time.perf_counter()
    n_batches = 0
    n_samples = 0

    for x, stamp, s1_ids, s2_ids, positions in loader:
        # full token = s1 * 1024 + s2
        ft_batch = (s1_ids * 1024 + s2_ids).numpy().flatten()
        all_full_tokens.extend(ft_batch.tolist())

        # per-window rare hit count
        ft_windows = (s1_ids * 1024 + s2_ids).numpy()   # (B, seq_len)
        for w in ft_windows:
            hits = sum(1 for t in w if t in rare_set_global)
            rare_hits_per_window.append(hits)

        n_batches += 1
        n_samples += x.shape[0]

    elapsed = time.perf_counter() - t0

    # Stats
    token_counts    = Counter(all_full_tokens)
    entropy_bits    = -sum(
        (c / len(all_full_tokens)) * np.log2(c / len(all_full_tokens))
        for c in token_counts.values()
    )
    rare_rate       = sum(1 for t in all_full_tokens if t in rare_set_global) / len(all_full_tokens)
    mean_rare_hits  = np.mean(rare_hits_per_window)
    pct_windows_with_rare = 100 * np.mean([h > 0 for h in rare_hits_per_window])
    top5_pct        = 100 * sum(v for _, v in token_counts.most_common(5)) / len(all_full_tokens)

    print(f"── {label} ──────────────────────────────────────")
    print(f"  Batches:          {n_batches}  |  Samples: {n_samples:,}  |  Tokens: {len(all_full_tokens):,}")
    print(f"  Throughput:       {n_batches/elapsed:.1f} batches/s  |  {n_samples/elapsed:.0f} samples/s")
    print(f"  Unique tokens:    {len(token_counts)}")
    print(f"  Token entropy:    {entropy_bits:.3f} bits  (higher = more variety)")
    print(f"  Rare-token rate:  {100*rare_rate:.2f}% of all token positions")
    print(f"  Windows w/ ≥1 rare token: {pct_windows_with_rare:.1f}%")
    print(f"  Mean rare hits/window:    {mean_rare_hits:.3f}")
    print(f"  Top-5 tokens %:   {top5_pct:.1f}%")
    print()

    return {
        'label':             label,
        'elapsed':           elapsed,
        'n_batches':         n_batches,
        'n_samples':         n_samples,
        'batches_per_sec':   n_batches / elapsed,
        'samples_per_sec':   n_samples / elapsed,
        'entropy':           entropy_bits,
        'rare_rate':         rare_rate,
        'pct_windows_rare':  pct_windows_with_rare,
        'mean_rare_hits':    mean_rare_hits,
        'top5_pct':          top5_pct,
        'token_counts':      token_counts,
        'rare_hits_per_window': rare_hits_per_window,
    }


# ── Run both loaders ──────────────────────────────────────────────────────────
print("Building Vanilla DataLoader…")
vanilla_loader = make_vanilla_loader(SEQ_LEN, BATCH_SIZE, STEPS_PER_EPOCH)

print("Building Roaring DataLoader…")
roaring_loader = make_roaring_loader(SEQ_LEN, BATCH_SIZE, STEPS_PER_EPOCH,
                                     RARE_THRESHOLD, SHOCK_FRAC)
print()

vanilla_stats = run_loader(vanilla_loader, "Vanilla (uniform)")
roaring_stats = run_loader(roaring_loader, f"Roaring (shock_frac={SHOCK_FRAC:.0%})")


# ── Head-to-head summary ──────────────────────────────────────────────────────
print("=" * 55)
print("HEAD-TO-HEAD COMPARISON")
print("=" * 55)
metrics = [
    ("Throughput (batches/s)", "batches_per_sec", "{:.1f}", True),
    ("Token entropy (bits)",   "entropy",         "{:.3f}", True),
    ("Rare-token rate (%)",    "rare_rate",        "{:.2%}", True),
    ("Windows w/ ≥1 rare (%)", "pct_windows_rare", "{:.1f}", True),
    ("Mean rare hits/window",  "mean_rare_hits",   "{:.3f}", True),
    ("Top-5 token dominance",  "top5_pct",         "{:.1f}%", False),
]
for name, key, fmt, higher_better in metrics:
    v = vanilla_stats[key]
    r = roaring_stats[key]
    v_str = fmt.format(v)
    r_str = fmt.format(r)
    winner = "Roaring ↑" if (r > v) == higher_better else "Vanilla"
    print(f"  {name:<32} Vanilla={v_str:<10} Roaring={r_str:<10} → {winner}")


# ── Plot ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Vanilla vs Roaring DataLoader — BTC 1h", fontsize=13, fontweight='bold')

# 1. Rare hits per window distribution
ax = axes[0]
max_hits = max(max(vanilla_stats['rare_hits_per_window']),
               max(roaring_stats['rare_hits_per_window']))
bins = np.arange(0, max_hits + 2) - 0.5
v_hist = np.histogram(vanilla_stats['rare_hits_per_window'], bins=bins)[0]
r_hist = np.histogram(roaring_stats['rare_hits_per_window'], bins=bins)[0]
x_ticks = np.arange(0, max_hits + 1)
width = 0.35
ax.bar(x_ticks - width/2, v_hist, width, label='Vanilla', color='steelblue', alpha=0.85)
ax.bar(x_ticks + width/2, r_hist, width, label='Roaring', color='coral',     alpha=0.85)
ax.set_title('Rare-token hits per window')
ax.set_xlabel('# rare tokens in window'); ax.set_ylabel('# windows')
ax.legend(); ax.grid(axis='y', alpha=0.4)

# 2. Token frequency rank curves (log-log)
ax = axes[1]
for stats, color, label in [
    (vanilla_stats, 'steelblue', 'Vanilla'),
    (roaring_stats, 'coral',     'Roaring'),
]:
    freqs = sorted(stats['token_counts'].values(), reverse=True)
    ax.plot(range(1, len(freqs)+1), freqs, color=color, label=label, linewidth=2)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title('Token frequency rank curve (log-log)')
ax.set_xlabel('Token rank'); ax.set_ylabel('Count')
ax.legend(); ax.grid(True, alpha=0.3)

# 3. Bar chart of key metrics
ax = axes[2]
metric_labels = ['Entropy\n(bits)', 'Rare rate\n(×100)', 'Windows\nw/ rare (%)']
v_vals = [
    vanilla_stats['entropy'],
    vanilla_stats['rare_rate'] * 100,
    vanilla_stats['pct_windows_rare'],
]
r_vals = [
    roaring_stats['entropy'],
    roaring_stats['rare_rate'] * 100,
    roaring_stats['pct_windows_rare'],
]
x = np.arange(len(metric_labels))
width = 0.35
ax.bar(x - width/2, v_vals, width, label='Vanilla', color='steelblue', alpha=0.85)
ax.bar(x + width/2, r_vals, width, label='Roaring', color='coral',     alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(metric_labels)
ax.set_title('Key quality metrics')
ax.legend(); ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight')
print(f"\nPlot saved → {PLOT_PATH}")
