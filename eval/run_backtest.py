"""
Generic single-checkpoint backtest runner.

Runs rolling-window autoregressive inference on the BTC test split
with a given Kronos checkpoint, computes the four signals
(last/mean/max/min) and the standard metrics (RankIC, CumRet, Sharpe,
MaxDD) for each, and saves a JSON.

Used by multi_seed_runner.py and shock_frac_sweep.py.

Run:
  PYENV_VERSION=3.10.14 python eval/run_backtest.py \\
    --checkpoint outputs/models/vanilla_finetuned/best_model \\
    --output     outputs/models/vanilla_finetuned/backtest.json
"""

import sys, os, argparse, pickle, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kronos_src'))

import numpy as np
import pandas as pd
import torch
from tqdm import trange
from scipy.stats import spearmanr

from model import KronosTokenizer, Kronos, KronosPredictor

LOOKBACK       = 90
PRED_LEN       = 10
SAMPLE_COUNT   = 5
T              = 0.6
TOP_P          = 0.9
CLIP           = 5.0
STRIDE         = 5
# Test region = the last (1 - TRAIN_FRAC) of the corpus. This boundary MUST
# match training/roaring_dataloader.py:VAL_END_FRAC so the fine-tune never
# trains or validates on any bar that the backtest scores.
TRAIN_FRAC     = 0.80
ROUND_TRIP_PCT = 0.0015
BARS_PER_YEAR  = 8760 // STRIDE


def rank_ic(signal: pd.Series, actual: pd.Series) -> float:
    df = pd.concat([signal, actual], axis=1).dropna()
    if len(df) < 5: return float('nan')
    rho, _ = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(rho)


def backtest_signal(signal: pd.Series, prices: pd.Series,
                    cost: float = ROUND_TRIP_PCT):
    position = (signal > 0).astype(float)
    asset_ret = prices.pct_change().shift(-1).reindex(signal.index)
    gross = position * asset_ret
    cost_paid = position.diff().abs() * cost
    return gross - cost_paid.fillna(0)


def compute_perf(net: pd.Series, bars_per_year: int = BARS_PER_YEAR) -> dict:
    cum = (1 + net.fillna(0)).cumprod()
    total = float(cum.iloc[-1] - 1)
    sharpe = float(net.mean() / (net.std() + 1e-10) * (bars_per_year ** 0.5))
    dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return {'total_return': total, 'sharpe': sharpe, 'max_drawdown': dd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to Kronos model checkpoint dir (or HF id)')
    parser.add_argument('--output', required=True,
                        help='Where to write the backtest results JSON')
    parser.add_argument('--data', default=None,
                        help='Path to OHLCVA CSV (defaults to data/btc_1h.csv)')
    parser.add_argument('--label', default='backtest',
                        help='Label string saved into the JSON')
    args = parser.parse_args()

    data_path = args.data or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'btc_1h.csv'
    )

    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    split = int(len(df) * TRAIN_FRAC)
    print(f"  Train: {split}  |  Test: {len(df)-split} bars")

    device = ("cuda" if torch.cuda.is_available() else
              "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
              "cpu")
    print(f"Device: {device}\n")

    print(f"Loading tokenizer + model from {args.checkpoint}…")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model     = Kronos.from_pretrained(args.checkpoint)
    predictor = KronosPredictor(model, tokenizer, device=device,
                                max_context=512, clip=CLIP)

    price_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    records = []
    for ctx_end in trange(split + LOOKBACK, len(df) - PRED_LEN, STRIDE,
                          desc=args.label, ncols=80):
        ctx = df.iloc[ctx_end - LOOKBACK:ctx_end]
        fut = df.iloc[ctx_end:ctx_end + PRED_LEN]
        pred = predictor.predict(
            df=ctx[price_cols].copy(),
            x_timestamp=pd.to_datetime(ctx['timestamp']),
            y_timestamp=pd.to_datetime(fut['timestamp']),
            pred_len=PRED_LEN, T=T, top_p=TOP_P,
            sample_count=SAMPLE_COUNT, verbose=False,
        )
        cur = ctx['close'].iloc[-1]
        records.append({
            'timestamp':   ctx['timestamp'].iloc[-1],
            'sig_last':    pred['close'].iloc[-1] - cur,
            'sig_mean':    pred['close'].mean() - cur,
            'sig_max':     pred['close'].max()  - cur,
            'sig_min':     pred['close'].min()  - cur,
            'actual_ret':  (fut['close'].iloc[-1] - cur) / cur,
            'current_close': cur,
        })

    r = pd.DataFrame(records).set_index('timestamp')

    # Compute metrics for each signal
    prices = df.set_index('timestamp')['close'].reindex(r.index)
    out = {'config': {'checkpoint': args.checkpoint, 'lookback': LOOKBACK,
                      'pred_len': PRED_LEN, 'stride': STRIDE, 'n_windows': len(r)}}
    for sig in ['sig_last', 'sig_mean', 'sig_max', 'sig_min']:
        rets = backtest_signal(r[sig], prices)
        perf = compute_perf(rets)
        perf['rank_ic'] = rank_ic(r[sig], r['actual_ret'])
        out[sig] = perf

    # Buy & hold benchmark
    bnh = compute_perf(prices.pct_change().reindex(r.index))
    out['benchmark'] = bnh

    # Save predictions as pickle (optional, for downstream analysis)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    pickle_path = args.output.replace('.json', '_predictions.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(r, f)

    print(f"\n✓ Saved {args.output}")
    print(f"✓ Saved {pickle_path}")
    print(f"\nHeadline (sig_last): IC={out['sig_last']['rank_ic']:+.4f}  "
          f"CumRet={out['sig_last']['total_return']:+.2%}  "
          f"Sharpe={out['sig_last']['sharpe']:+.3f}  "
          f"MaxDD={out['sig_last']['max_drawdown']:+.2%}")


if __name__ == '__main__':
    main()
