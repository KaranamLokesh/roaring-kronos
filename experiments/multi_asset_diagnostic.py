"""
Multi-asset replication of the hierarchical co-occurrence diagnostic.

Tests whether the BTC findings (extreme vocabulary sparsity, 52/163
deterministic-s2 coarse tokens) hold on other asset classes:
  - ETH-USD  (alt crypto)
  - SPY      (US equity)

Output:
  experiments/results/multi_asset_diagnostic.json  — full numbers
  experiments/multi_asset_diagnostic.png            — comparison plot

Paper deliverable:
  Updates Table 1 (vocab utilization) with ETH and SPY columns,
  updates Table 3 (cooc diagnostic) with ETH and SPY entries.

Run (~20-30 min total):
  PYENV_VERSION=3.10.14 python experiments/multi_asset_diagnostic.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kronos_src'))

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from collections import Counter
from pyroaring import BitMap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import KronosTokenizer

CLIP        = 5.0
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
PLOT_PATH   = os.path.join(os.path.dirname(__file__), 'multi_asset_diagnostic.png')


# ── Step 1: data ──────────────────────────────────────────────────────────────
def fetch_bars(ticker: str, period: str = '2y', interval: str = '1h') -> pd.DataFrame:
    """yfinance hourly OHLCV → DataFrame with [open, high, low, close, volume, amount]."""
    csv_path = os.path.join(DATA_DIR, f'{ticker.replace("-","_").replace("=","_").lower()}_1h.csv')
    if os.path.exists(csv_path):
        print(f"  cached: {csv_path}")
        return pd.read_csv(csv_path, parse_dates=['timestamp'])

    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    raw = raw.dropna()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw['amount'] = raw['volume'] * raw[['open', 'high', 'low', 'close']].mean(axis=1)
    df = raw[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    df.index.name = 'timestamp'
    df = df.reset_index()
    df.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")
    return df


# ── Step 2: tokenize ──────────────────────────────────────────────────────────
def tokenize(df: pd.DataFrame, tokenizer, device):
    """Returns (s1_tokens, s2_tokens) numpy arrays."""
    cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    x = df[cols].values.astype(np.float32)
    mean, std = x.mean(axis=0), x.std(axis=0)
    x_norm = np.clip((x - mean) / (std + 1e-5), -CLIP, CLIP)

    with torch.no_grad():
        x_t = torch.from_numpy(x_norm).unsqueeze(0).to(device)
        idx = tokenizer.encode(x_t, half=True)
        s1 = idx[0].squeeze(0).cpu().numpy()
        s2 = idx[1].squeeze(0).cpu().numpy()
    return s1, s2


# ── Step 3: diagnostics ───────────────────────────────────────────────────────
def diagnose(s1: np.ndarray, s2: np.ndarray, s1_bits: int = 10, s2_bits: int = 10) -> dict:
    """Run the full diagnostic suite for one asset."""
    N = len(s1)
    s2_vocab = 2 ** s2_bits

    # 1. Vocabulary utilization
    s1_unique = sorted(set(s1.tolist()))
    s2_unique = sorted(set(s2.tolist()))
    full = (s1 * s2_vocab + s2)
    full_unique = sorted(set(full.tolist()))

    # 2. Entropy
    def entropy(arr):
        c = Counter(arr.tolist())
        p = np.array([v / len(arr) for v in c.values()])
        return -np.sum(p * np.log2(p + 1e-10))

    # 3. Frequency concentration
    full_counts = Counter(full.tolist())
    sorted_freqs = np.array(sorted(full_counts.values(), reverse=True))
    cumsum = np.cumsum(sorted_freqs) / len(full)
    top10_pct  = 100 * cumsum[min(9, len(cumsum)-1)]
    p80_rank   = int(np.searchsorted(cumsum, 0.80)) + 1

    # 4. s1 run-length distribution
    runs = []
    cur, run_len = s1[0], 1
    for t in s1[1:]:
        if t == cur:
            run_len += 1
        else:
            runs.append(run_len)
            cur = t; run_len = 1
    runs.append(run_len)
    runs = np.array(runs)

    # 5. Hierarchical co-occurrence
    cooc = {}
    for s1v, s2v in zip(s1.tolist(), s2.tolist()):
        cooc.setdefault(s1v, set()).add(s2v)

    cooc_sizes = [len(v) for v in cooc.values()]
    deterministic = sum(1 for v in cooc.values() if len(v) == 1)

    return {
        'n_bars': int(N),
        's1_used':  len(s1_unique),
        's1_total': 2 ** s1_bits,
        's2_used':  len(s2_unique),
        's2_total': 2 ** s2_bits,
        'full_used':  len(full_unique),
        'full_total': 2 ** (s1_bits + s2_bits),
        'vocab_utilization_pct': 100 * len(full_unique) / (2 ** (s1_bits + s2_bits)),
        's1_entropy_bits':   float(entropy(s1)),
        's2_entropy_bits':   float(entropy(s2)),
        'full_entropy_bits': float(entropy(full)),
        'top10_pct_coverage':   float(top10_pct),
        'p80_rank':             int(p80_rank),
        's1_run_mean':          float(runs.mean()),
        's1_run_median':        float(np.median(runs)),
        's1_run_max':           int(runs.max()),
        's1_run_pct_ge3':       float(100 * (runs >= 3).sum() / len(runs)),
        'active_s1_count':      len(cooc),
        'deterministic_s1_count': deterministic,
        'deterministic_s1_pct':   100 * deterministic / len(cooc),
        'mean_s2_per_s1':       float(np.mean(cooc_sizes)),
        'median_s2_per_s1':     float(np.median(cooc_sizes)),
        'max_s2_per_s1':        int(max(cooc_sizes)),
    }


def main():
    print("=" * 66)
    print("Multi-asset hierarchical diagnostic")
    print("=" * 66)

    device = ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else "cpu")
    print(f"\nLoading Kronos-Tokenizer-base on {device}…")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    tokenizer = tokenizer.to(device).eval()

    assets = [
        ('BTC-USD', 'BTC'),
        ('ETH-USD', 'ETH'),
        ('SPY',     'SPY'),
    ]
    results = {}

    for ticker, label in assets:
        print(f"\n── {label} ({ticker}) ─────────────────────────")
        df = fetch_bars(ticker, period='2y', interval='1h')
        print(f"  {len(df)} bars  |  {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        s1, s2 = tokenize(df, tokenizer, device)
        results[label] = diagnose(s1, s2)
        d = results[label]
        print(f"  Vocab utilization: {d['vocab_utilization_pct']:.3f}%  "
              f"({d['full_used']}/{d['full_total']})")
        print(f"  Top-49 coverage:   {d['p80_rank']} tokens cover 80%")
        print(f"  s1 run mean/max:   {d['s1_run_mean']:.2f} / {d['s1_run_max']}")
        print(f"  Active s1:         {d['active_s1_count']}")
        print(f"  Deterministic s1:  {d['deterministic_s1_count']} "
              f"({d['deterministic_s1_pct']:.1f}% of active)")
        print(f"  Mean s2 per s1:    {d['mean_s2_per_s1']:.2f}")

    # ── Save JSON ────────────────────────────────────────────────────────
    out_json = os.path.join(RESULTS_DIR, 'multi_asset_diagnostic.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved {out_json}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Hierarchical BSQ diagnostic across asset classes", fontsize=13, fontweight='bold')
    labels = list(results.keys())
    colors = ['steelblue', 'coral', 'mediumseagreen']

    # 1. vocab utilization (log)
    ax = axes[0, 0]
    ax.bar(labels, [results[l]['vocab_utilization_pct'] for l in labels], color=colors)
    ax.set_ylabel('% of vocab used'); ax.set_title('Full vocabulary utilization')
    for i, l in enumerate(labels):
        ax.text(i, results[l]['vocab_utilization_pct'],
                f"{results[l]['vocab_utilization_pct']:.3f}%",
                ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 2. p80 rank
    ax = axes[0, 1]
    ax.bar(labels, [results[l]['p80_rank'] for l in labels], color=colors)
    ax.set_ylabel('# tokens'); ax.set_title('Tokens to reach 80% coverage')
    for i, l in enumerate(labels):
        ax.text(i, results[l]['p80_rank'], str(results[l]['p80_rank']),
                ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 3. deterministic s1
    ax = axes[0, 2]
    det_pct = [results[l]['deterministic_s1_pct'] for l in labels]
    ax.bar(labels, det_pct, color=colors)
    ax.set_ylabel('% of active s1'); ax.set_title('Active s1 with deterministic s2')
    for i, l in enumerate(labels):
        ax.text(i, det_pct[i], f"{det_pct[i]:.1f}%",
                ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 4. mean s2 per s1
    ax = axes[1, 0]
    ax.bar(labels, [results[l]['mean_s2_per_s1'] for l in labels], color=colors)
    ax.set_ylabel('mean |Cooc(s1)|'); ax.set_title('Mean fine tokens per coarse')
    ax.grid(axis='y', alpha=0.3)

    # 5. run-length mean
    ax = axes[1, 1]
    ax.bar(labels, [results[l]['s1_run_mean'] for l in labels], color=colors)
    ax.set_ylabel('mean run length (bars)'); ax.set_title('s1 regime persistence')
    ax.grid(axis='y', alpha=0.3)

    # 6. full entropy
    ax = axes[1, 2]
    ax.bar(labels, [results[l]['full_entropy_bits'] for l in labels], color=colors)
    ax.axhline(20, color='red', linestyle='--', alpha=0.6, label='Max (20.0)')
    ax.set_ylabel('bits'); ax.set_title('Full token entropy')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight')
    print(f"✓ Saved {PLOT_PATH}")

    # ── Print paper-ready comparison table ─────────────────────────────
    print("\n" + "=" * 66)
    print("PAPER-READY COMPARISON TABLE")
    print("=" * 66)
    print(f"{'Metric':<32}", end="")
    for l in labels: print(f"{l:>10}", end="")
    print()
    print("-" * 66)
    metrics = [
        ('Bars', 'n_bars', '{:>10d}'),
        ('Vocab utilization (%)', 'vocab_utilization_pct', '{:>10.3f}'),
        ('p80 token rank', 'p80_rank', '{:>10d}'),
        ('s1 entropy (bits)', 's1_entropy_bits', '{:>10.2f}'),
        ('Full entropy (bits)', 'full_entropy_bits', '{:>10.2f}'),
        ('s1 run mean', 's1_run_mean', '{:>10.2f}'),
        ('s1 run max', 's1_run_max', '{:>10d}'),
        ('Active s1', 'active_s1_count', '{:>10d}'),
        ('Deterministic s1', 'deterministic_s1_count', '{:>10d}'),
        ('Deterministic s1 (%)', 'deterministic_s1_pct', '{:>10.1f}'),
        ('Mean |Cooc(s1)|', 'mean_s2_per_s1', '{:>10.2f}'),
        ('Max |Cooc(s1)|', 'max_s2_per_s1', '{:>10d}'),
    ]
    for label, key, fmt in metrics:
        print(f"{label:<32}", end="")
        for l in labels:
            print(fmt.format(results[l][key]), end="")
        print()


if __name__ == '__main__':
    main()
