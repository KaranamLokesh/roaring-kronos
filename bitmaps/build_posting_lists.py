"""
Build Roaring Bitmap posting lists over the tokenized BTC corpus and run
three experiments:

  1. Compression ratio  — Roaring vs raw uint32 list per token
  2. Rare-event lookup  — find all "shock" bars instantly
  3. Hierarchical diagnostic — s2 coverage per s1 context + container type audit

Run from repo root:
  PYENV_VERSION=3.10.14 python bitmaps/build_posting_lists.py
"""

import os, sys, time, struct
import numpy as np
from pyroaring import BitMap, FrozenBitMap
from collections import defaultdict, Counter

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'bitmaps')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load tokens ───────────────────────────────────────────────────────────────
print("Loading tokens…")
s1 = np.load(os.path.join(DATA_DIR, 'btc_1h_s1_tokens.npy'))   # (T,) int64
s2 = np.load(os.path.join(DATA_DIR, 'btc_1h_s2_tokens.npy'))
ft = np.load(os.path.join(DATA_DIR, 'btc_1h_full_tokens.npy'))
T  = len(ft)
print(f"  {T} bars  |  s1_vocab={s1.max()+1}  s2_vocab={s2.max()+1}  full_range=[{ft.min()},{ft.max()}]")

s1_uniq = sorted(set(s1.tolist()))
s2_uniq = sorted(set(s2.tolist()))
ft_uniq = sorted(set(ft.tolist()))


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Build posting lists & measure compression
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 1 — Posting list compression (Roaring vs raw uint32)")
print("="*70)

# Build s1 posting lists
t0 = time.perf_counter()
s1_bitmaps: dict[int, BitMap] = {}
for pos, tok in enumerate(s1.tolist()):
    if tok not in s1_bitmaps:
        s1_bitmaps[tok] = BitMap()
    s1_bitmaps[tok].add(pos)
s1_build_ms = (time.perf_counter() - t0) * 1000
print(f"\n  s1 posting lists built in {s1_build_ms:.1f} ms  ({len(s1_bitmaps)} bitmaps)")

# Build full-token posting lists
t0 = time.perf_counter()
ft_bitmaps: dict[int, BitMap] = {}
for pos, tok in enumerate(ft.tolist()):
    if tok not in ft_bitmaps:
        ft_bitmaps[tok] = BitMap()
    ft_bitmaps[tok].add(pos)
ft_build_ms = (time.perf_counter() - t0) * 1000
print(f"  Full-token posting lists built in {ft_build_ms:.1f} ms  ({len(ft_bitmaps)} bitmaps)")

# Compression stats
def compression_stats(bitmaps: dict[int, BitMap], label: str):
    roaring_bytes = 0
    raw_bytes     = 0
    ratios        = []
    for bm in bitmaps.values():
        rb = len(bm.serialize())
        raw = len(bm) * 4          # uint32 per position
        roaring_bytes += rb
        raw_bytes     += raw
        if raw > 0:
            ratios.append(raw / rb)
    ratio_overall = raw_bytes / roaring_bytes if roaring_bytes else 1.0
    print(f"\n  [{label}]")
    print(f"    Raw uint32 total:    {raw_bytes:>10,} bytes  ({raw_bytes/1024:.1f} KB)")
    print(f"    Roaring total:       {roaring_bytes:>10,} bytes  ({roaring_bytes/1024:.1f} KB)")
    print(f"    Overall ratio:       {ratio_overall:.2f}×")
    print(f"    Per-bitmap ratios — min={min(ratios):.2f}×  median={sorted(ratios)[len(ratios)//2]:.2f}×  max={max(ratios):.2f}×")
    return ratio_overall

r1 = compression_stats(s1_bitmaps,  "s1 coarse tokens")
r2 = compression_stats(ft_bitmaps, "full tokens")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Rare-event lookup
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 2 — Rare-event lookup speed")
print("="*70)

# Identify "rare" full tokens (appear ≤ 5 times) and "common" ones (top 5)
ft_counts = Counter(ft.tolist())
rare_tokens  = [tok for tok, cnt in ft_counts.items() if cnt <= 5]
common_tokens = [tok for tok, cnt in ft_counts.most_common(5)]

print(f"\n  Rare tokens (count ≤ 5): {len(rare_tokens)}")
print(f"  Common tokens (top-5):   {len(common_tokens)}")

# Baseline: linear scan for rare events
t0 = time.perf_counter()
rare_set = set(rare_tokens)
baseline_positions = [i for i, tok in enumerate(ft.tolist()) if tok in rare_set]
baseline_ms = (time.perf_counter() - t0) * 1000

# Roaring: union of rare token bitmaps
t0 = time.perf_counter()
rare_union = BitMap()
for tok in rare_tokens:
    rare_union |= ft_bitmaps[tok]
roaring_ms = (time.perf_counter() - t0) * 1000

assert sorted(rare_union.to_array().tolist()) == sorted(baseline_positions), "Results mismatch!"

print(f"\n  Linear scan:      {baseline_ms:.2f} ms  → {len(baseline_positions)} positions")
print(f"  Roaring union:    {roaring_ms:.2f} ms  → {len(rare_union)} positions")
print(f"  Speedup:          {baseline_ms/roaring_ms:.1f}×")

# What fraction of bars are "shocks" vs "chop"?
top5_positions = BitMap()
for tok in common_tokens:
    top5_positions |= ft_bitmaps[tok]

shock_count = len(rare_union)
chop_count  = len(top5_positions)
print(f"\n  Top-5 token bars (chop):  {chop_count:5d}  ({100*chop_count/T:.1f}% of data)")
print(f"  Rare-token bars (shocks): {shock_count:5d}  ({100*shock_count/T:.1f}% of data)")

# Demonstrate stratified sampling: interleave chop + shock for training
t0 = time.perf_counter()
# 50/50 mix: pick equal amounts from chop and shock posting lists
chop_sample_size  = min(500, chop_count)
shock_sample_size = min(500, shock_count)
chop_arr  = np.array(top5_positions.to_array())
shock_arr = np.array(rare_union.to_array())
rng = np.random.default_rng(42)
chop_sample  = rng.choice(chop_arr,  chop_sample_size,  replace=False)
shock_sample = rng.choice(shock_arr, shock_sample_size, replace=False)
batch = np.sort(np.concatenate([chop_sample, shock_sample]))
strat_ms = (time.perf_counter() - t0) * 1000
print(f"\n  Stratified batch assembly (50/50 chop+shock, n={len(batch)}): {strat_ms:.2f} ms")
print(f"  → Positions sample: {batch[:10].tolist()} …")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3 — Hierarchical diagnostic
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 3 — Hierarchical diagnostic (s2 co-occurrence per s1)")
print("="*70)

# For each active s1 value, build a bitmap over the s2 vocabulary (size=1024)
# showing which fine tokens actually co-occur with it.
s2_vocab = int(s2.max()) + 1   # observed max+1

cooc_bitmaps: dict[int, BitMap] = {}
for pos, (s1v, s2v) in enumerate(zip(s1.tolist(), s2.tolist())):
    if s1v not in cooc_bitmaps:
        cooc_bitmaps[s1v] = BitMap()
    cooc_bitmaps[s1v].add(s2v)

coverages   = []
single_s2   = []   # s1 values with only 1 fine token
tight_s1    = []   # coverage < 5%
active_s1   = sorted(cooc_bitmaps.keys())

for s1v in active_s1:
    n = len(cooc_bitmaps[s1v])
    coverages.append(n)
    if n == 1:
        single_s2.append(s1v)
    if n / s2_vocab < 0.05:
        tight_s1.append(s1v)

print(f"\n  Active s1 values: {len(active_s1)}")
print(f"  s1 values with ONLY 1 fine token:   {len(single_s2)}  (perfectly tight hierarchy)")
print(f"  s1 values with < 5% s2 coverage:    {len(tight_s1)}")
print(f"  Mean s2 tokens per s1:   {np.mean(coverages):.2f}")
print(f"  Median:                  {np.median(coverages):.0f}")
print(f"  Max:                     {max(coverages)}")

# Container type analysis — what Roaring container does each s1 cooc bitmap use?
# Roaring uses: Array (<= 4096 set bits, small), Bitmap (dense), Run (run-encoded)
# For small bitsets over s2_vocab=1024, everything fits in Array containers.
# The interesting metric is: are the s2 values that co-occur with each s1
# *contiguous* (clustered in integer space) or scattered?
# Proxy: for each s1, measure the "span" (max_s2 - min_s2 + 1) vs actual count.
# span == count means perfectly contiguous → Run Container territory.
# span >> count means scattered → Array Container.

spans    = []
densities = []
for s1v in active_s1:
    bm = cooc_bitmaps[s1v]
    arr = sorted(bm.to_array())
    span = arr[-1] - arr[0] + 1 if len(arr) > 1 else 1
    spans.append(span)
    densities.append(len(arr) / span)

print(f"\n  Container analysis (s2 integer-space clustering):")
print(f"  Mean density (count/span):  {np.mean(densities):.3f}  (1.0=perfectly contiguous, ~0=scattered)")
print(f"  Median density:             {np.median(densities):.3f}")
low_density = sum(1 for d in densities if d < 0.1)
print(f"  s1 values with density < 0.1 (scattered): {low_density} / {len(active_s1)}")
print(f"  → Scattered s2 patterns mean the BSQ bit-ordering is not semantically aligned")
print(f"    (expected — BSQ assigns no meaning to bit position order)")

# Show tightest + loosest examples
idx_sort = sorted(range(len(active_s1)), key=lambda i: coverages[i])
print(f"\n  Tightest 5 s1 values (fewest s2 partners):")
for i in idx_sort[:5]:
    s1v = active_s1[i]
    bm  = cooc_bitmaps[s1v]
    cnt = s1_bitmaps[s1v]
    print(f"    s1={s1v:4d}  s2_tokens={sorted(bm.to_array())}  occurrences={len(cnt)}")

print(f"\n  Loosest 5 s1 values (most s2 partners):")
for i in idx_sort[-5:]:
    s1v = active_s1[i]
    bm  = cooc_bitmaps[s1v]
    cnt = s1_bitmaps[s1v]
    print(f"    s1={s1v:4d}  s2_count={len(bm)}  occurrences={len(cnt)}")


# ── Serialize all bitmaps to disk ─────────────────────────────────────────────
print("\n" + "="*70)
print("Serializing bitmaps to disk…")

import pickle
payload = {
    's1_bitmaps':   {k: bytes(v.serialize()) for k, v in s1_bitmaps.items()},
    'ft_bitmaps':   {k: bytes(v.serialize()) for k, v in ft_bitmaps.items()},
    'cooc_bitmaps': {k: bytes(v.serialize()) for k, v in cooc_bitmaps.items()},
    'meta': {
        'T': T,
        's1_vocab_observed': len(s1_bitmaps),
        's2_vocab_observed': len(s2_uniq),
        'ft_vocab_observed': len(ft_bitmaps),
        's1_bits': 10,
        's2_bits': 10,
    }
}
out_path = os.path.join(OUT_DIR, 'btc_1h_bitmaps.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(payload, f)

size_kb = os.path.getsize(out_path) / 1024
print(f"  Saved to data/bitmaps/btc_1h_bitmaps.pkl  ({size_kb:.1f} KB)")

# Compare: what would raw numpy arrays cost?
raw_total = sum(len(bm) * 4 for bm in s1_bitmaps.values())
raw_total += sum(len(bm) * 4 for bm in ft_bitmaps.values())
raw_total += sum(len(bm) * 4 for bm in cooc_bitmaps.values())
print(f"  Equivalent raw uint32:  {raw_total/1024:.1f} KB")
print(f"  Roaring savings:        {raw_total/1024 - size_kb:.1f} KB  ({100*(1 - size_kb*1024/raw_total):.1f}% smaller)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Corpus:           {T:,} BTC 1h bars")
print(f"  Vocab coverage:   417 / 1,048,576 full tokens  (0.04%)")
print(f"  s1 compression:   {r1:.2f}× vs raw uint32")
print(f"  Full compression: {r2:.2f}× vs raw uint32")
print(f"  Rare-event speedup: {baseline_ms/roaring_ms:.1f}× over linear scan")
print(f"  Single-s2 s1 values: {len(single_s2)} (perfectly tight hierarchy)")
print(f"  Roaring bitmap store: {size_kb:.1f} KB for all posting lists")
