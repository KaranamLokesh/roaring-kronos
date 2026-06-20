"""
RoaringDataLoader — stratified window sampler for Kronos pretraining.

Instead of picking random starting positions uniformly, it uses prebuilt
Roaring Bitmap posting lists to oversample windows that contain rare/shock
tokens, so the model sees crash/spike bars proportionally more often.

Usage:
    from training.roaring_dataloader import RoaringDataLoader, VanillaDataLoader
"""

import os, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
from pyroaring import BitMap

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
BITMAP_DIR = os.path.join(DATA_DIR, 'bitmaps')

# Chronological split — MUST stay in lockstep with eval/run_backtest.py:TRAIN_FRAC.
# Training and validation windows occupy [0, VAL_END_FRAC); the backtest scores
# only [VAL_END_FRAC, 1.0] (the last 20%). No bar from the test region is ever
# sampled during fine-tuning, which eliminates train/test leakage.
TRAIN_END_FRAC = 0.70   # [0.00, 0.70)  → training windows
VAL_END_FRAC   = 0.80   # [0.70, 0.80)  → validation windows  (== backtest test boundary)
# [0.80, 1.00) is the backtest test region — never sampled here.


def _region_bounds(split, T):
    """Return [lo, hi) bar bounds for the requested split. None → full corpus."""
    if split is None:
        return 0, T
    if split == 'train':
        return 0, int(T * TRAIN_END_FRAC)
    if split == 'val':
        return int(T * TRAIN_END_FRAC), int(T * VAL_END_FRAC)
    if split == 'test':
        return int(T * VAL_END_FRAC), T
    raise ValueError(f"Unknown split: {split!r}")


def _load_corpus():
    """Load raw OHLCVA + precomputed token arrays from disk."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'btc_1h.csv'), parse_dates=['timestamp'])
    s1 = np.load(os.path.join(DATA_DIR, 'btc_1h_s1_tokens.npy'))
    s2 = np.load(os.path.join(DATA_DIR, 'btc_1h_s2_tokens.npy'))
    ft = np.load(os.path.join(DATA_DIR, 'btc_1h_full_tokens.npy'))
    return df, s1, s2, ft


def _normalize(x: np.ndarray, clip: float = 5.0) -> np.ndarray:
    mean = x.mean(axis=0)
    std  = x.std(axis=0)
    return np.clip((x - mean) / (std + 1e-5), -clip, clip).astype(np.float32)


def _time_features(timestamps: pd.Series) -> np.ndarray:
    df = pd.DataFrame()
    df['minute']  = timestamps.dt.minute
    df['hour']    = timestamps.dt.hour
    df['weekday'] = timestamps.dt.weekday
    df['day']     = timestamps.dt.day
    df['month']   = timestamps.dt.month
    return df.values.astype(np.float32)


class KronosWindowDataset(IterableDataset):
    """
    Base iterable dataset. Subclasses supply `_sample_positions()` to
    choose which windows to yield each epoch.

    Each yielded item:
        x          (seq_len, 6)   normalized OHLCVA
        stamp      (seq_len, 5)   time features
        s1_ids     (seq_len,)     coarse token ids
        s2_ids     (seq_len,)     fine token ids
        positions  (seq_len,)     corpus positions (for debugging)
    """

    def __init__(self, seq_len: int = 128, steps_per_epoch: int = 512,
                 seed: int = 42, split: str | None = None):
        super().__init__()
        self.seq_len         = seq_len
        self.steps_per_epoch = steps_per_epoch
        self.rng             = np.random.default_rng(seed)
        self.split           = split

        df, self.s1, self.s2, self.ft = _load_corpus()
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.x_norm  = _normalize(df[price_cols].values)
        self.stamps  = _time_features(df['timestamp'])
        self.T       = len(self.ft)

        # Restrict sampleable window starts to one chronological region so that
        # a window [start, start+seq_len) never spills past the region boundary
        # (and in particular never into the held-out backtest test period).
        self.region_lo, self.region_hi = _region_bounds(split, self.T)
        self.start_lo      = self.region_lo
        self.start_hi_excl = self.region_hi - seq_len + 1   # exclusive upper bound
        if self.start_hi_excl <= self.start_lo:
            raise ValueError(
                f"Region [{self.region_lo},{self.region_hi}) too small for "
                f"seq_len={seq_len} (split={split!r})")
        self.max_start = self.start_hi_excl - 1   # back-compat alias

    def _sample_positions(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def __iter__(self):
        starts = self._sample_positions(self.steps_per_epoch)
        # Shard across DataLoader workers so num_workers>1 does not duplicate
        # windows (each worker holds an identical RNG-seeded copy of self).
        info = torch.utils.data.get_worker_info()
        if info is not None:
            starts = starts[info.id::info.num_workers]
        for start in starts:
            end = start + self.seq_len
            yield (
                torch.from_numpy(self.x_norm[start:end]),
                torch.from_numpy(self.stamps[start:end]),
                torch.from_numpy(self.s1[start:end].astype(np.int64)),
                torch.from_numpy(self.s2[start:end].astype(np.int64)),
                torch.arange(start, end, dtype=torch.long),
            )


class VanillaDataset(KronosWindowDataset):
    """Uniform random window sampling — the baseline."""

    def _sample_positions(self, n: int) -> np.ndarray:
        return self.rng.integers(self.start_lo, self.start_hi_excl, size=n)


class RoaringDataset(KronosWindowDataset):
    """
    Stratified window sampling using Roaring Bitmap posting lists.

    Strategy:
      - shock_frac of windows are anchored at a rare-token position
        (the rare token appears somewhere within the window)
      - remaining windows are sampled uniformly (chop)

    rare_threshold: full tokens with count <= this value are "rare"
    shock_frac:     fraction of each batch that comes from shock anchors
    """

    def __init__(
        self,
        seq_len:         int   = 128,
        steps_per_epoch: int   = 512,
        seed:            int   = 42,
        split:           str | None = None,
        rare_threshold:  int   = 5,
        shock_frac:      float = 0.3,
    ):
        super().__init__(seq_len, steps_per_epoch, seed, split=split)
        self.shock_frac = shock_frac

        with open(os.path.join(BITMAP_DIR, 'btc_1h_bitmaps.pkl'), 'rb') as f:
            store = pickle.load(f)

        # Rebuild ft bitmaps
        from collections import Counter
        ft_counts = Counter(self.ft.tolist())
        rare_tokens = [tok for tok, cnt in ft_counts.items() if cnt <= rare_threshold]

        ft_bitmaps = {
            k: BitMap.deserialize(v) for k, v in store['ft_bitmaps'].items()
        }

        # Union of all rare-token positions
        rare_union = BitMap()
        for tok in rare_tokens:
            if tok in ft_bitmaps:
                rare_union |= ft_bitmaps[tok]

        # Valid anchor positions: restricted to this split's region so the
        # anchored window [p, p+seq_len) stays inside [region_lo, region_hi).
        self.shock_anchors = np.array([
            p for p in rare_union.to_array()
            if self.start_lo <= p < self.start_hi_excl
        ], dtype=np.int64)

        # Chop positions: region complement of the shock anchors
        region_positions = np.arange(self.start_lo, self.start_hi_excl, dtype=np.int64)
        if len(self.shock_anchors):
            self.chop_positions = region_positions[
                ~np.isin(region_positions, self.shock_anchors)]
        else:
            self.chop_positions = region_positions

        print(f"  [RoaringDataset] split={split} region=[{self.region_lo},{self.region_hi}) "
              f"| shock anchors: {len(self.shock_anchors)} "
              f"| chop pool: {len(self.chop_positions)} "
              f"| shock_frac={shock_frac:.0%}")

    def _sample_positions(self, n: int) -> np.ndarray:
        # No rare tokens in this region → fall back to uniform sampling.
        if len(self.shock_anchors) == 0:
            return self.rng.integers(self.start_lo, self.start_hi_excl, size=n)
        n_shock = int(n * self.shock_frac)
        n_chop  = n - n_shock

        shock_idx = self.rng.choice(len(self.shock_anchors), size=n_shock, replace=True)
        chop_idx  = self.rng.choice(len(self.chop_positions),  size=n_chop,  replace=True)

        shock_starts = self.shock_anchors[shock_idx]
        chop_starts  = self.chop_positions[chop_idx]

        starts = np.concatenate([shock_starts, chop_starts])
        self.rng.shuffle(starts)
        return starts


def make_vanilla_loader(seq_len=128, batch_size=32, steps_per_epoch=512) -> DataLoader:
    ds = VanillaDataset(seq_len=seq_len, steps_per_epoch=steps_per_epoch)
    return DataLoader(ds, batch_size=batch_size)


def make_roaring_loader(
    seq_len=128, batch_size=32, steps_per_epoch=512,
    rare_threshold=5, shock_frac=0.3
) -> DataLoader:
    ds = RoaringDataset(
        seq_len=seq_len,
        steps_per_epoch=steps_per_epoch,
        rare_threshold=rare_threshold,
        shock_frac=shock_frac,
    )
    return DataLoader(ds, batch_size=batch_size)
