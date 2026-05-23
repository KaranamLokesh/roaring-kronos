"""
BTC Kronos Backtest — zero-shot baseline using Kronos-small.

Mirrors Kronos's official Qlib evaluation protocol:
  lookback = 90 bars, pred_len = 10 bars, T=0.6, top_p=0.9, sample_count=5

Produces 4 signals (last / mean / max / min predicted close delta) and reports:
  - RankIC (rank correlation of predicted vs actual next-bar return)
  - Cumulative return with 0.15% round-trip cost
  - Sharpe ratio (annualised, 8760 hourly bars/year)
  - Max drawdown
  - vs buy-and-hold BTC benchmark

Run from repo root:
  PYENV_VERSION=3.10.14 python eval/btc_backtest.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kronos_src'))

import numpy as np
import pandas as pd
import torch
from tqdm import trange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from model import KronosTokenizer, Kronos, KronosPredictor

# ── Config (mirrors Kronos finetune/config.py) ────────────────────────────────
LOOKBACK       = 90
PRED_LEN       = 10
SAMPLE_COUNT   = 5
T              = 0.6
TOP_P          = 0.9
CLIP           = 5.0
STRIDE         = 5       # evaluate every STRIDE bars (faster; 1 = full rolling)
TRAIN_FRAC     = 0.80    # first 80 % = training region (not touched)
ROUND_TRIP_PCT = 0.0015  # 0.15% round-trip transaction cost

DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'btc_1h.csv')
RESULT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'experiments')
RESULT_PKL  = os.path.join(RESULT_DIR, 'btc_backtest_baseline.pkl')
PLOT_PATH   = os.path.join(RESULT_DIR, 'btc_backtest_baseline.png')
os.makedirs(RESULT_DIR, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading BTC 1h data…")
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

split = int(len(df) * TRAIN_FRAC)
test_df = df.iloc[split:].reset_index(drop=True)
print(f"  Train: {split} bars | Test: {len(test_df)} bars "
      f"({test_df['timestamp'].iloc[0].date()} → {test_df['timestamp'].iloc[-1].date()})")

price_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
# Use full df for context (model can look back into train set)
full_x = df[price_cols].values.astype(np.float32)


# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading Kronos-small…")
device = "mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu"
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512, clip=CLIP)
print(f"  Device: {device}")


# ── Rolling inference ─────────────────────────────────────────────────────────
print(f"\nRolling inference — stride={STRIDE}, lookback={LOOKBACK}, pred_len={PRED_LEN}…")

# Test windows start at split + LOOKBACK so each has full context
# Context window always drawn from full df (can reach into train)
window_starts = range(split + LOOKBACK, len(df) - PRED_LEN, STRIDE)
n_windows = len(window_starts)
print(f"  Windows to evaluate: {n_windows}")

records = []   # (context_end_idx, signal_last, signal_mean, signal_max, signal_min, actual_return)

for i, ctx_end in enumerate(trange(split + LOOKBACK, len(df) - PRED_LEN, STRIDE,
                                    desc="Inference", ncols=80)):
    ctx_start = ctx_end - LOOKBACK
    fut_end   = ctx_end + PRED_LEN

    ctx_slice = df.iloc[ctx_start:ctx_end]
    fut_slice = df.iloc[ctx_end:fut_end]

    x_df        = ctx_slice[price_cols].copy()
    x_timestamps = pd.to_datetime(ctx_slice['timestamp'])
    y_timestamps = pd.to_datetime(fut_slice['timestamp'])

    pred_df = predictor.predict(
        df          = x_df,
        x_timestamp = x_timestamps,
        y_timestamp = y_timestamps,
        pred_len    = PRED_LEN,
        T           = T,
        top_p       = TOP_P,
        sample_count= SAMPLE_COUNT,
        verbose     = False,
    )

    current_close = ctx_slice['close'].iloc[-1]
    actual_close  = fut_slice['close'].iloc[-1]

    # Signals = predicted close (various aggregations) - current close
    sig_last = pred_df['close'].iloc[-1]  - current_close
    sig_mean = pred_df['close'].mean()    - current_close
    sig_max  = pred_df['close'].max()     - current_close
    sig_min  = pred_df['close'].min()     - current_close

    actual_ret = (actual_close - current_close) / current_close

    records.append({
        'ctx_end':    ctx_end,
        'timestamp':  ctx_slice['timestamp'].iloc[-1],
        'sig_last':   sig_last,
        'sig_mean':   sig_mean,
        'sig_max':    sig_max,
        'sig_min':    sig_min,
        'actual_ret': actual_ret,
        'current_close': current_close,
    })

results = pd.DataFrame(records).set_index('timestamp')
print(f"\n  Generated {len(results)} prediction records")

import pickle
with open(RESULT_PKL, 'wb') as f:
    pickle.dump(results, f)
print(f"  Saved raw predictions → {RESULT_PKL}")


# ── Metrics ───────────────────────────────────────────────────────────────────
def rank_ic(pred_col: str, df: pd.DataFrame) -> float:
    """Spearman rank correlation of signal vs actual return (single asset = IC over time)."""
    valid = df[[pred_col, 'actual_ret']].dropna()
    if len(valid) < 5:
        return float('nan')
    rho, _ = spearmanr(valid[pred_col], valid['actual_ret'])
    return rho


def backtest_signal(signal: pd.Series, prices: pd.Series,
                    threshold: float = 0.0, cost: float = ROUND_TRIP_PCT):
    """
    Simple long-only strategy: go long if signal > threshold, flat otherwise.
    Returns: series of per-step net returns.
    """
    position = (signal > threshold).astype(float)
    # Actual return of the asset over each prediction window
    asset_ret = prices.pct_change().shift(-1).reindex(signal.index)

    gross = position * asset_ret
    # Pay cost on position changes
    cost_paid = position.diff().abs() * cost
    net = gross - cost_paid.fillna(0)
    return net


def compute_perf(net_rets: pd.Series, bars_per_year: int = 8760 // 5):
    """Compute annualised Sharpe, cumulative return, max drawdown."""
    cum = (1 + net_rets.fillna(0)).cumprod()
    total_ret = cum.iloc[-1] - 1

    ann_factor = bars_per_year ** 0.5
    sharpe = net_rets.mean() / (net_rets.std() + 1e-10) * ann_factor

    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd   = drawdown.min()

    return {'total_return': total_ret, 'sharpe': sharpe, 'max_drawdown': max_dd, 'cum': cum}


# Align prices to results index
prices = df.set_index('timestamp')['close'].reindex(results.index)

print("\n" + "="*65)
print("BACKTEST RESULTS — Zero-shot Kronos-small on BTC 1h (test set)")
print("="*65)

signal_cols = ['sig_last', 'sig_mean', 'sig_max', 'sig_min']
perf_records = {}

for col in signal_cols:
    ic   = rank_ic(col, results)
    rets = backtest_signal(results[col], prices)
    perf = compute_perf(rets)
    perf['ic'] = ic
    perf_records[col] = perf

    print(f"\n  Signal: {col}")
    print(f"    RankIC:          {ic:+.4f}")
    print(f"    Cumulative ret:  {perf['total_return']:+.2%}")
    print(f"    Sharpe (ann):    {perf['sharpe']:+.3f}")
    print(f"    Max drawdown:    {perf['max_drawdown']:+.2%}")

# Buy-and-hold benchmark
bnh_rets  = prices.pct_change().reindex(results.index)
bnh_perf  = compute_perf(bnh_rets)
print(f"\n  Benchmark (Buy & Hold BTC)")
print(f"    Cumulative ret:  {bnh_perf['total_return']:+.2%}")
print(f"    Sharpe (ann):    {bnh_perf['sharpe']:+.3f}")
print(f"    Max drawdown:    {bnh_perf['max_drawdown']:+.2%}")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Kronos-small Zero-Shot Backtest — BTC-USD 1h (baseline)", fontsize=13, fontweight='bold')

colors = ['steelblue', 'coral', 'mediumseagreen', 'orchid']

# Cumulative returns
ax = axes[0, 0]
for col, color in zip(signal_cols, colors):
    rets = backtest_signal(results[col], prices)
    cum  = (1 + rets.fillna(0)).cumprod()
    ax.plot(cum.index, cum.values, label=col.replace('sig_', ''), color=color, linewidth=1.5)
bnh_cum = (1 + bnh_rets.fillna(0)).cumprod()
ax.plot(bnh_cum.index, bnh_cum.values, 'k--', label='Buy & Hold', linewidth=1.5, alpha=0.7)
ax.set_title('Cumulative Return (with 0.15% cost)'); ax.set_ylabel('Cumul. return')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Excess return vs B&H
ax = axes[0, 1]
for col, color in zip(signal_cols, colors):
    rets = backtest_signal(results[col], prices)
    cum  = (1 + rets.fillna(0)).cumprod()
    excess = cum / bnh_cum.reindex(cum.index).fillna(method='ffill')
    ax.plot(excess.index, excess.values, label=col.replace('sig_', ''), color=color, linewidth=1.5)
ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
ax.set_title('Cumulative Excess Return vs B&H'); ax.set_ylabel('Ratio vs benchmark')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# RankIC bar chart
ax = axes[1, 0]
ic_vals  = [perf_records[col]['ic'] for col in signal_cols]
ic_colors = ['green' if v > 0 else 'red' for v in ic_vals]
ax.bar([c.replace('sig_','') for c in signal_cols], ic_vals, color=ic_colors, alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('RankIC (Spearman, signal vs actual return)'); ax.set_ylabel('RankIC')
ax.grid(axis='y', alpha=0.3)

# Sharpe bar chart
ax = axes[1, 1]
sharpe_vals = [perf_records[col]['sharpe'] for col in signal_cols]
ax.bar([c.replace('sig_','') for c in signal_cols], sharpe_vals, color=colors, alpha=0.8)
ax.axhline(bnh_perf['sharpe'], color='black', linestyle='--', alpha=0.7, label='B&H Sharpe')
ax.set_title('Annualised Sharpe Ratio'); ax.set_ylabel('Sharpe')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight')
print(f"\nPlot saved → {PLOT_PATH}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY TABLE (baseline — zero-shot Kronos-small)")
print("="*65)
print(f"{'Signal':<12} {'RankIC':>8} {'CumRet':>10} {'Sharpe':>8} {'MaxDD':>10}")
print("-"*55)
for col in signal_cols:
    p = perf_records[col]
    print(f"{col.replace('sig_',''):<12} {p['ic']:>+8.4f} {p['total_return']:>+9.2%} "
          f"{p['sharpe']:>+8.3f} {p['max_drawdown']:>+9.2%}")
print(f"{'B&H':<12} {'N/A':>8} {bnh_perf['total_return']:>+9.2%} "
      f"{bnh_perf['sharpe']:>+8.3f} {bnh_perf['max_drawdown']:>+9.2%}")
print("\nThis is the baseline to beat after Roaring-augmented fine-tuning.")
