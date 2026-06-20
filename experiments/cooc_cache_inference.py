"""
cooc-cache inference optimization — measures the inference speedup
from skipping the s2 decode for deterministic s1 tokens.

Mechanism: for each s1 value v where |Cooc(v)| == 1, the fine-token
prediction is fully determined: s2 = the single value in Cooc(v).
We cache this map and short-circuit the autoregressive decode loop.

At inference Kronos decodes each bar in two stages (kronos.py):
  1. decode_s1  → s1 logits + transformer context
  2. decode_s2(context, s1)  → s2 logits, via the dependency-aware
     cross-attention layer + the conditional s2 head
For a deterministic s1 we can skip stage 2 entirely and emit the
cached s2. This script times the REAL model.decode_s2 call (not a
mock) on both paths and reports the measured speedup at the empirical
rate at which a corpus bar lands on a deterministic s1.

Output:
  experiments/results/cooc_cache_inference.json   — latency & coverage
  experiments/cooc_cache_inference.png             — comparison plot

Paper deliverable: concrete inference-time win cited in §4.6 and §7.

Run (~10-15 min on MacBook MPS / a few min on A100):
  PYENV_VERSION=3.10.14 python experiments/cooc_cache_inference.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kronos_src'))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import KronosTokenizer, Kronos

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(REPO_ROOT, 'data')
RESULTS_DIR = os.path.join(REPO_ROOT, 'experiments', 'results')
PLOT_PATH   = os.path.join(REPO_ROOT, 'experiments', 'cooc_cache_inference.png')
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_cooc_cache(corpus_dir: str) -> dict:
    """Build the s1 → s2 deterministic-lookup table from corpus data."""
    s1 = np.load(os.path.join(corpus_dir, 'btc_1h_s1_tokens.npy'))
    s2 = np.load(os.path.join(corpus_dir, 'btc_1h_s2_tokens.npy'))

    cooc = {}
    for s1v, s2v in zip(s1.tolist(), s2.tolist()):
        cooc.setdefault(s1v, set()).add(s2v)

    deterministic = {v: next(iter(partners))
                     for v, partners in cooc.items() if len(partners) == 1}
    return {
        'deterministic': deterministic,
        'cooc_sizes':    {v: len(p) for v, p in cooc.items()},
        'n_active_s1':   len(cooc),
        'n_deterministic': len(deterministic),
        's1_distribution': {v: int((s1 == v).sum()) for v in cooc},
    }


def _time_features(ts: pd.Series) -> np.ndarray:
    """[minute, hour, weekday, day, month] — matches TemporalEmbedding order."""
    out = pd.DataFrame()
    out['minute']  = ts.dt.minute
    out['hour']    = ts.dt.hour
    out['weekday'] = ts.dt.weekday
    out['day']     = ts.dt.day
    out['month']   = ts.dt.month
    return out.values.astype(np.float32)


def build_real_context(model, device, batch: int, seq_len: int):
    """
    Assemble a real (context, s1_ids) pair by running the model's own
    decode_s1 over genuine corpus windows. The returned context is what
    decode_s2 consumes per step at inference time.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, 'btc_1h.csv'), parse_dates=['timestamp'])
    s1 = np.load(os.path.join(DATA_DIR, 'btc_1h_s1_tokens.npy'))
    s2 = np.load(os.path.join(DATA_DIR, 'btc_1h_s2_tokens.npy'))
    stamps = _time_features(df['timestamp'])

    # Draw `batch` windows from the held-out tail of the corpus.
    rng = np.random.default_rng(0)
    hi = len(s1) - seq_len
    starts = rng.integers(int(hi * 0.8), hi, size=batch)
    s1_ids = np.stack([s1[p:p + seq_len] for p in starts]).astype(np.int64)
    s2_ids = np.stack([s2[p:p + seq_len] for p in starts]).astype(np.int64)
    stamp  = np.stack([stamps[p:p + seq_len] for p in starts]).astype(np.float32)

    s1_t = torch.from_numpy(s1_ids).to(device)
    s2_t = torch.from_numpy(s2_ids).to(device)
    st_t = torch.from_numpy(stamp).to(device)
    with torch.no_grad():
        _, context = model.decode_s1(s1_t, s2_t, st_t)
    return context, s1_t


def benchmark_inference(model_id: str = 'NeoQuasar/Kronos-small',
                        n_steps: int = 300, batch: int = 32, seq_len: int = 90):
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else "cpu")
    print(f"Device: {device}")

    # Build cache from corpus
    cache = build_cooc_cache(DATA_DIR)
    det_map = cache['deterministic']
    print(f"Active s1: {cache['n_active_s1']}, deterministic: {cache['n_deterministic']} "
          f"({100*cache['n_deterministic']/cache['n_active_s1']:.1f}%)")

    # Probability that a random corpus bar lands on a deterministic s1
    s1_dist = cache['s1_distribution']
    total = sum(s1_dist.values())
    det_bar_prob = sum(c for v, c in s1_dist.items() if v in det_map) / total
    print(f"P(deterministic) over a random corpus bar: {det_bar_prob:.1%}")

    # Load the real model and a real context for decode_s2
    print(f"\nLoading {model_id}…")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device).eval()
    model = Kronos.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    context, s1_ids = build_real_context(model, device, batch, seq_len)
    print(f"Real context built: {tuple(context.shape)}")

    def sync():
        if device == 'cuda': torch.cuda.synchronize()
        elif device == 'mps': torch.mps.synchronize()

    # Sample which s1 each step decodes, from the empirical distribution
    rng = np.random.default_rng(42)
    s1_vals  = list(s1_dist.keys())
    s1_probs = np.array([s1_dist[v] for v in s1_vals], dtype=np.float64)
    s1_probs /= s1_probs.sum()
    step_s1 = rng.choice(s1_vals, size=n_steps, p=s1_probs)
    det_set = set(det_map.keys())

    def one_s2_decode():
        with torch.no_grad():
            s2_logits = model.decode_s2(context, s1_ids)
            _ = s2_logits[:, -1, :]   # the bar being generated

    # ── Baseline: real decode_s2 on every step ──
    print(f"\nBenchmarking baseline (real decode_s2 every step)…")
    for _ in range(5):
        one_s2_decode()
    sync()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        one_s2_decode()
    sync()
    baseline_s = time.perf_counter() - t0
    baseline_ms = baseline_s / n_steps * 1000
    print(f"  total: {baseline_s:.2f}s  ({baseline_ms:.3f} ms/step)")

    # ── Cached: skip decode_s2 when the step's s1 is deterministic ──
    print(f"\nBenchmarking cooc-cache (skip decode_s2 for deterministic s1)…")
    for _ in range(5):
        one_s2_decode()
    sync()
    t0 = time.perf_counter()
    n_skipped = 0
    for v in step_s1:
        if v in det_set:
            _ = det_map[v]      # O(1) cache hit — no model call
            n_skipped += 1
        else:
            one_s2_decode()
    sync()
    cached_s = time.perf_counter() - t0
    cached_ms = cached_s / n_steps * 1000
    print(f"  total: {cached_s:.2f}s  ({cached_ms:.3f} ms/step)  "
          f"| skipped {n_skipped}/{n_steps} steps ({100*n_skipped/n_steps:.1f}%)")

    speedup = baseline_s / cached_s

    results = {
        'device': str(device),
        'model': model_id,
        'n_steps': n_steps,
        'batch_size': batch,
        'seq_len': seq_len,
        'active_s1':        cache['n_active_s1'],
        'deterministic_s1': cache['n_deterministic'],
        'deterministic_pct': 100 * cache['n_deterministic'] / cache['n_active_s1'],
        'det_bar_pct':      100 * det_bar_prob,
        'steps_skipped':    int(n_skipped),
        'baseline_ms_per_step': baseline_ms,
        'cached_ms_per_step':   cached_ms,
        'baseline_total_s': baseline_s,
        'cached_total_s':   cached_s,
        'speedup':          speedup,
    }

    out_json = os.path.join(RESULTS_DIR, 'cooc_cache_inference.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved {out_json}")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('cooc-cache inference optimisation (real decode_s2)',
                 fontsize=12, fontweight='bold')

    bars = ax1.bar(['baseline', 'cooc-cache'], [baseline_ms, cached_ms],
                   color=['steelblue', 'coral'])
    for b, v in zip(bars, [baseline_ms, cached_ms]):
        ax1.text(b.get_x() + b.get_width()/2, v, f'{v:.2f} ms',
                 ha='center', va='bottom', fontweight='bold')
    ax1.set_ylabel('ms per step')
    ax1.set_title(f'decode_s2 latency  ({speedup:.2f}× speedup)')
    ax1.grid(axis='y', alpha=0.3)

    stats = [cache['n_active_s1'], cache['n_deterministic'],
             cache['n_active_s1'] - cache['n_deterministic']]
    ax2.bar(['active s1', 'deterministic', 'non-deterministic'],
            stats, color=['gray', 'green', 'orange'])
    for i, v in enumerate(stats):
        ax2.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    ax2.set_ylabel('# s1 values')
    ax2.set_title(f'Cache coverage  ({100*det_bar_prob:.1f}% of bars hit cache)')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight')
    print(f"✓ Saved {PLOT_PATH}")

    print("\n" + "=" * 60)
    print(f"  Baseline:    {baseline_ms:.3f} ms/step (real decode_s2)")
    print(f"  Cooc-cache:  {cached_ms:.3f} ms/step")
    print(f"  Speedup:     {speedup:.2f}×")
    print(f"  Cache hit:   {det_bar_prob:.1%} of bars skip decode_s2")
    print("=" * 60)


if __name__ == '__main__':
    benchmark_inference()
