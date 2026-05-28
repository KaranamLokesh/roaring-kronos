"""
cooc-cache inference optimization — measures the inference speedup
from skipping the s2 head for deterministic s1 tokens.

Mechanism: for each s1 value v where |Cooc(v)| == 1, the fine-token
prediction is fully determined: s2 = the single value in Cooc(v).
We cache this map and short-circuit the autoregressive decode loop.

For non-deterministic s1, we fall back to the normal s2 head softmax.

Output:
  experiments/results/cooc_cache_inference.json   — latency & memory
  experiments/cooc_cache_inference.png             — comparison plot

Paper deliverable: concrete inference-time win cited in §4.6 and §7.

Run (~15 min on MacBook MPS):
  PYENV_VERSION=3.10.14 python experiments/cooc_cache_inference.py
"""

import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kronos_src'))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pyroaring import BitMap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import KronosTokenizer, Kronos

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def time_decode_baseline(model, s1_logits, top_p=0.9, T=0.6):
    """Standard sampling: argmax/sample from s2 head logits."""
    s2_logits = model.head.proj_s2(torch.randn_like(
        s1_logits.new_zeros(s1_logits.size(0), 1024)))  # mock activation
    probs = F.softmax(s2_logits / T, dim=-1)
    return torch.multinomial(probs, 1)


def benchmark_inference(model_path: str = None, n_steps: int = 200, batch: int = 32):
    """
    Compares two decode strategies:
      baseline:  always run s2 head softmax over 1024 classes
      cached:    if s1 in deterministic map, return cached s2; else fall back

    We simulate by drawing s1 values from the empirical corpus distribution.
    """
    device = ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
              else "cpu")
    print(f"Device: {device}")

    # Build cache from corpus
    corpus_dir = os.path.join(REPO_ROOT, 'data')
    cache = build_cooc_cache(corpus_dir)
    det_map = cache['deterministic']
    print(f"Active s1: {cache['n_active_s1']}, deterministic: {cache['n_deterministic']} "
          f"({100*cache['n_deterministic']/cache['n_active_s1']:.1f}%)")

    # Probability that a random bar hits a deterministic s1
    s1_dist = cache['s1_distribution']
    total = sum(s1_dist.values())
    det_bar_prob = sum(c for v, c in s1_dist.items() if v in det_map) / total
    print(f"P(deterministic) over a random corpus bar: {det_bar_prob:.1%}")

    # Sample s1 values from corpus distribution
    rng = np.random.default_rng(42)
    s1_values = list(s1_dist.keys())
    s1_probs  = np.array([s1_dist[v] for v in s1_values], dtype=np.float64)
    s1_probs /= s1_probs.sum()

    # Generate test sequence of s1 tokens
    sampled_s1 = rng.choice(s1_values, size=n_steps * batch, p=s1_probs)

    # ── Baseline: run s2 head softmax for every token ──
    print(f"\nBenchmarking baseline (s2 head every step)…")
    s2_proj = torch.nn.Linear(256, 1024).to(device).eval()
    activation = torch.randn(batch, 256, device=device)

    for _ in range(5):  # warmup
        with torch.no_grad():
            _ = F.softmax(s2_proj(activation) / 0.6, dim=-1)
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        with torch.no_grad():
            logits = s2_proj(activation)
            probs  = F.softmax(logits / 0.6, dim=-1)
            sample = torch.multinomial(probs, 1)
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    baseline_s = time.perf_counter() - t0
    baseline_ms_per_step = baseline_s / n_steps * 1000
    print(f"  total: {baseline_s:.2f}s  ({baseline_ms_per_step:.3f} ms/step)")

    # ── Cached: check map first, fall back to softmax otherwise ──
    print(f"\nBenchmarking cached (skip s2 head for deterministic s1)…")
    # Build a tensor mask: which s1 values are deterministic, and their s2
    det_keys = sorted(det_map.keys())
    max_s1 = max(s1_values) + 1
    cache_tensor = torch.full((max_s1,), -1, dtype=torch.long, device=device)
    for k, v in det_map.items():
        cache_tensor[k] = v
    print(f"  Cache size: {max_s1} entries  ({max_s1 * 8} bytes)")

    # warmup
    for _ in range(5):
        with torch.no_grad():
            batch_s1 = torch.from_numpy(sampled_s1[:batch]).to(device)
            cached_s2 = cache_tensor[batch_s1]
            mask = cached_s2 == -1
            if mask.any():
                logits = s2_proj(activation[mask])
                probs  = F.softmax(logits / 0.6, dim=-1)
                _ = torch.multinomial(probs, 1)
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    idx = 0
    for _ in range(n_steps):
        with torch.no_grad():
            batch_s1 = torch.from_numpy(sampled_s1[idx:idx+batch]).to(device)
            cached_s2 = cache_tensor[batch_s1]
            mask = cached_s2 == -1
            if mask.any():
                logits = s2_proj(activation[mask])
                probs  = F.softmax(logits / 0.6, dim=-1)
                sample = torch.multinomial(probs, 1)
        idx += batch
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    cached_s = time.perf_counter() - t0
    cached_ms_per_step = cached_s / n_steps * 1000
    print(f"  total: {cached_s:.2f}s  ({cached_ms_per_step:.3f} ms/step)")

    speedup = baseline_s / cached_s

    # ── Save results ─────────────────────────────────────────────────
    results = {
        'device': str(device),
        'n_steps': n_steps,
        'batch_size': batch,
        'active_s1':       cache['n_active_s1'],
        'deterministic_s1': cache['n_deterministic'],
        'deterministic_pct': 100 * cache['n_deterministic'] / cache['n_active_s1'],
        'det_bar_pct':     100 * det_bar_prob,
        'baseline_ms_per_step': baseline_ms_per_step,
        'cached_ms_per_step':   cached_ms_per_step,
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
    fig.suptitle('cooc-cache inference optimisation', fontsize=12, fontweight='bold')

    # latency
    bars = ax1.bar(['baseline', 'cooc-cache'],
                   [baseline_ms_per_step, cached_ms_per_step],
                   color=['steelblue', 'coral'])
    for b, v in zip(bars, [baseline_ms_per_step, cached_ms_per_step]):
        ax1.text(b.get_x() + b.get_width()/2, v, f'{v:.2f} ms',
                 ha='center', va='bottom', fontweight='bold')
    ax1.set_ylabel('ms per step')
    ax1.set_title(f's2-head decode latency  ({speedup:.2f}× speedup)')
    ax1.grid(axis='y', alpha=0.3)

    # cache statistics
    stats = [cache['n_active_s1'],
             cache['n_deterministic'],
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
    print(f"  Baseline:    {baseline_ms_per_step:.3f} ms/step")
    print(f"  Cooc-cache:  {cached_ms_per_step:.3f} ms/step")
    print(f"  Speedup:     {speedup:.2f}×")
    print(f"  Cache hit:   {det_bar_prob:.1%} of bars skip s2 head")
    print("=" * 60)


if __name__ == '__main__':
    benchmark_inference()
