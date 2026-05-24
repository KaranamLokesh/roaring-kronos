"""
Run the BTC backtest on all three model variants:
  1. zero-shot Kronos-small  (already done — loads from baseline pickle)
  2. vanilla-finetuned       (uniform sampling)
  3. roaring-finetuned       (30% shock anchored)

Generates side-by-side comparison plots and a summary table.

Run from repo root:
  PYENV_VERSION=3.10.14 python eval/btc_backtest_finetuned.py
"""

import sys, os, pickle
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

# ── Match baseline backtest config exactly ────────────────────────────────────
LOOKBACK       = 90
PRED_LEN       = 10
SAMPLE_COUNT   = 5
T              = 0.6
TOP_P          = 0.9
CLIP           = 5.0
STRIDE         = 5
TRAIN_FRAC     = 0.80
ROUND_TRIP_PCT = 0.0015

DATA_PATH    = os.path.join(os.path.dirname(__file__), '..', 'data', 'btc_1h.csv')
RESULT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'experiments')
MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')
BASELINE_PKL = os.path.join(RESULT_DIR, 'btc_backtest_baseline.pkl')
PLOT_PATH    = os.path.join(RESULT_DIR, 'btc_backtest_comparison.png')


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading BTC 1h data…")
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

split = int(len(df) * TRAIN_FRAC)
print(f"  Train: {split} bars | Test: {len(df)-split} bars")

price_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']


def run_inference(model_label: str, model_path: str | None) -> pd.DataFrame:
    """Run rolling inference and return predictions DataFrame."""
    print(f"\n── Running inference: {model_label} ──")

    device = "mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu"
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    if model_path is None:
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    else:
        model = Kronos.from_pretrained(model_path)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512, clip=CLIP)

    records = []
    for ctx_end in trange(split + LOOKBACK, len(df) - PRED_LEN, STRIDE,
                          desc=f"  {model_label}", ncols=80):
        ctx_start = ctx_end - LOOKBACK
        fut_end   = ctx_end + PRED_LEN
        ctx_slice = df.iloc[ctx_start:ctx_end]
        fut_slice = df.iloc[ctx_end:fut_end]

        pred_df = predictor.predict(
            df          = ctx_slice[price_cols].copy(),
            x_timestamp = pd.to_datetime(ctx_slice['timestamp']),
            y_timestamp = pd.to_datetime(fut_slice['timestamp']),
            pred_len    = PRED_LEN,
            T           = T, top_p=TOP_P, sample_count=SAMPLE_COUNT,
            verbose     = False,
        )

        current_close = ctx_slice['close'].iloc[-1]
        actual_close  = fut_slice['close'].iloc[-1]
        records.append({
            'timestamp':  ctx_slice['timestamp'].iloc[-1],
            'sig_last':   pred_df['close'].iloc[-1] - current_close,
            'sig_mean':   pred_df['close'].mean()   - current_close,
            'sig_max':    pred_df['close'].max()    - current_close,
            'sig_min':    pred_df['close'].min()    - current_close,
            'actual_ret': (actual_close - current_close) / current_close,
            'current_close': current_close,
        })

    return pd.DataFrame(records).set_index('timestamp')


# ── Metrics ───────────────────────────────────────────────────────────────────
def rank_ic(pred_col, df):
    valid = df[[pred_col, 'actual_ret']].dropna()
    if len(valid) < 5: return float('nan')
    rho, _ = spearmanr(valid[pred_col], valid['actual_ret'])
    return rho


def backtest_signal(signal, prices, cost=ROUND_TRIP_PCT):
    position = (signal > 0).astype(float)
    asset_ret = prices.pct_change().shift(-1).reindex(signal.index)
    gross = position * asset_ret
    cost_paid = position.diff().abs() * cost
    return gross - cost_paid.fillna(0)


def compute_perf(net_rets, bars_per_year=8760 // 5):
    cum = (1 + net_rets.fillna(0)).cumprod()
    total_ret = cum.iloc[-1] - 1
    sharpe = net_rets.mean() / (net_rets.std() + 1e-10) * (bars_per_year ** 0.5)
    drawdown = (cum - cum.cummax()) / cum.cummax()
    return {'total_return': total_ret, 'sharpe': sharpe,
            'max_drawdown': drawdown.min(), 'cum': cum}


# ── Load or compute baseline ──────────────────────────────────────────────────
print("\nLoading zero-shot baseline predictions…")
with open(BASELINE_PKL, 'rb') as f:
    baseline_results = pickle.load(f)
print(f"  Loaded {len(baseline_results)} predictions")

# ── Run inference on fine-tuned models ────────────────────────────────────────
vanilla_path = os.path.join(MODELS_DIR, 'vanilla_finetuned', 'best_model')
roaring_path = os.path.join(MODELS_DIR, 'roaring_finetuned', 'best_model')

vanilla_results = run_inference("vanilla-finetuned", vanilla_path)
roaring_results = run_inference("roaring-finetuned", roaring_path)

# Save predictions
with open(os.path.join(RESULT_DIR, 'btc_backtest_vanilla.pkl'), 'wb') as f:
    pickle.dump(vanilla_results, f)
with open(os.path.join(RESULT_DIR, 'btc_backtest_roaring.pkl'), 'wb') as f:
    pickle.dump(roaring_results, f)

# ── Compute metrics on all three ──────────────────────────────────────────────
results_all = {
    'zero-shot': baseline_results,
    'vanilla-ft': vanilla_results,
    'roaring-ft': roaring_results,
}

signal_cols = ['sig_last', 'sig_mean', 'sig_max', 'sig_min']
metrics_all = {}

print("\n" + "="*75)
print("FULL COMPARISON — Zero-shot vs Vanilla-finetuned vs Roaring-finetuned")
print("="*75)

for variant, r in results_all.items():
    prices = df.set_index('timestamp')['close'].reindex(r.index)
    metrics_all[variant] = {}
    print(f"\n── {variant} ──")
    for col in signal_cols:
        ic   = rank_ic(col, r)
        rets = backtest_signal(r[col], prices)
        perf = compute_perf(rets)
        perf['ic'] = ic
        perf['rets'] = rets
        metrics_all[variant][col] = perf
        print(f"  {col:<10} IC={ic:+.4f}  CumRet={perf['total_return']:+.2%}  "
              f"Sharpe={perf['sharpe']:+.3f}  MaxDD={perf['max_drawdown']:+.2%}")


# ── Monthly IC breakdown — the key paper finding ─────────────────────────────
print("\n" + "="*75)
print("MONTHLY RANKIC (sig_last) — does Roaring help in the crash months?")
print("="*75)
print(f"  {'Month':<10} {'Zero-shot':>10} {'Vanilla-ft':>12} {'Roaring-ft':>12}")
print("  " + "-"*46)
all_months = pd.PeriodIndex(baseline_results.index.to_period('M').unique(), freq='M').sort_values()
for month in all_months:
    row = f"  {str(month):<10}"
    for variant, r in results_all.items():
        sub = r[r.index.to_period('M') == month]
        if len(sub) < 5:
            row += f"  {'n/a':>10}"
            continue
        ic, _ = spearmanr(sub['sig_last'], sub['actual_ret'])
        row += f"  {ic:>+10.3f}"
    print(row)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
fig.suptitle("Kronos Backtest: Zero-shot vs Vanilla-FT vs Roaring-FT (BTC 1h)",
             fontsize=14, fontweight='bold')

colors = {'zero-shot': 'gray', 'vanilla-ft': 'steelblue', 'roaring-ft': 'coral'}

# Cumulative returns — sig_last only (most representative signal)
ax = fig.add_subplot(gs[0, :])
prices_full = df.set_index('timestamp')['close'].reindex(baseline_results.index)
bnh_cum = (1 + prices_full.pct_change().fillna(0)).cumprod()
ax.plot(bnh_cum.index, bnh_cum.values, 'k--', label='Buy & Hold BTC',
        linewidth=1.5, alpha=0.7)
for variant, r in results_all.items():
    perf = metrics_all[variant]['sig_last']
    cum = (1 + perf['rets'].fillna(0)).cumprod()
    ax.plot(cum.index, cum.values, label=variant, color=colors[variant], linewidth=2)
ax.set_title('Cumulative Return — sig_last signal (with 0.15% cost)')
ax.set_ylabel('Cumul. return'); ax.legend(fontsize=10); ax.grid(alpha=0.3)

# RankIC by variant (bar chart)
ax = fig.add_subplot(gs[1, 0])
variants = list(results_all.keys())
ic_data = {col: [metrics_all[v][col]['ic'] for v in variants] for col in signal_cols}
x = np.arange(len(signal_cols))
width = 0.27
for i, v in enumerate(variants):
    vals = [metrics_all[v][col]['ic'] for col in signal_cols]
    ax.bar(x + (i-1)*width, vals, width, label=v, color=colors[v], alpha=0.85)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels([c.replace('sig_','') for c in signal_cols])
ax.set_title('RankIC by signal type'); ax.set_ylabel('RankIC')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

# Sharpe by variant
ax = fig.add_subplot(gs[1, 1])
for i, v in enumerate(variants):
    vals = [metrics_all[v][col]['sharpe'] for col in signal_cols]
    ax.bar(x + (i-1)*width, vals, width, label=v, color=colors[v], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([c.replace('sig_','') for c in signal_cols])
ax.set_title('Annualised Sharpe by signal type'); ax.set_ylabel('Sharpe')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

# Monthly IC heatmap
ax = fig.add_subplot(gs[2, :])
months_str = [str(m) for m in all_months]
heatmap_data = []
for v in variants:
    row = []
    for month in all_months:
        sub = results_all[v][results_all[v].index.to_period('M') == month]
        if len(sub) < 5:
            row.append(0)
        else:
            ic, _ = spearmanr(sub['sig_last'], sub['actual_ret'])
            row.append(ic)
    heatmap_data.append(row)
heatmap_data = np.array(heatmap_data)
im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.2)
ax.set_xticks(range(len(months_str))); ax.set_xticklabels(months_str)
ax.set_yticks(range(len(variants))); ax.set_yticklabels(variants)
ax.set_title('Monthly RankIC (sig_last) — green = predictive, red = anti-predictive')
for i in range(len(variants)):
    for j in range(len(months_str)):
        ax.text(j, i, f'{heatmap_data[i,j]:+.2f}', ha='center', va='center',
                color='black', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.025)

plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight')
print(f"\nComparison plot saved → {PLOT_PATH}")

# ── Final summary table ───────────────────────────────────────────────────────
print("\n" + "="*75)
print("HEADLINE NUMBERS — sig_last signal")
print("="*75)
print(f"{'Variant':<14} {'RankIC':>8} {'CumRet':>10} {'Sharpe':>8} {'MaxDD':>10}")
print("-"*55)
for v in variants:
    m = metrics_all[v]['sig_last']
    print(f"{v:<14} {m['ic']:>+8.4f} {m['total_return']:>+9.2%} "
          f"{m['sharpe']:>+8.3f} {m['max_drawdown']:>+9.2%}")
# Delta row
delta_ic = metrics_all['roaring-ft']['sig_last']['ic'] - metrics_all['vanilla-ft']['sig_last']['ic']
delta_ret = metrics_all['roaring-ft']['sig_last']['total_return'] - metrics_all['vanilla-ft']['sig_last']['total_return']
print(f"{'roaring - vanilla':<14} {delta_ic:>+8.4f} {delta_ret:>+9.2%}")
