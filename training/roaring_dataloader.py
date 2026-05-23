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

    def __init__(self, seq_len: int = 128, steps_per_epoch: int = 512, seed: int = 42):
        super().__init__()
        self.seq_len         = seq_len
        self.steps_per_epoch = steps_per_epoch
        self.rng             = np.random.default_rng(seed)

        df, self.s1, self.s2, self.ft = _load_corpus()
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.x_norm  = _normalize(df[price_cols].values)
        self.stamps  = _time_features(df['timestamp'])
        self.T       = len(self.ft)
        self.max_start = self.T - seq_len

    def _sample_positions(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def __iter__(self):
        starts = self._sample_positions(self.steps_per_epoch)
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
        return self.rng.integers(0, self.max_start, size=n)


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
        rare_threshold:  int   = 5,
        shock_frac:      float = 0.3,
    ):
        super().__init__(seq_len, steps_per_epoch, seed)
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

        # Valid anchor positions: rare token must not be too close to corpus end
        self.shock_anchors = np.array([
            p for p in rare_union.to_array() if p <= self.max_start
        ], dtype=np.int64)

        # Chop positions: complement — not near a shock
        all_positions = np.arange(self.max_start + 1, dtype=np.int64)
        shock_set = set(self.shock_anchors.tolist())
        self.chop_positions = all_positions[
            [i for i in range(len(all_positions)) if i not in shock_set]
        ]

        print(f"  [RoaringDataset] shock anchors: {len(self.shock_anchors)} "
              f"| chop pool: {len(self.chop_positions)} "
              f"| shock_frac={shock_frac:.0%}")

    def _sample_positions(self, n: int) -> np.ndarray:
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
