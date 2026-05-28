"""
shock_frac sweep — finds the optimal shock-anchoring fraction.

For each f in {0.15, 0.30, 0.45, 0.60}, runs the Roaring fine-tune
+ backtest and reports the headline metrics. Also runs vanilla (f=0)
as a baseline for context.

Output:
  experiments/results/shock_frac_sweep.json   — full metrics
  experiments/shock_frac_sweep.png             — sensitivity plot

Paper deliverable: new figure + table in §4.5 showing how the
choice of f trades off batch quality vs backtest performance.

Run (~2 hours on MacBook MPS):
  PYENV_VERSION=3.10.14 python experiments/shock_frac_sweep.py
"""

import sys, os, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, 'experiments', 'results')
PLOT_PATH   = os.path.join(REPO_ROOT, 'experiments', 'shock_frac_sweep.png')
os.makedirs(RESULTS_DIR, exist_ok=True)

SHOCK_FRACS = [0.0, 0.15, 0.30, 0.45, 0.60]
SEED   = 42
EPOCHS = 5


def run_one(shock_frac: float) -> dict:
    if shock_frac == 0.0:
        sampler  = 'vanilla'
        run_name = f'vanilla_seed{SEED}_finetuned'
    else:
        sampler  = 'roaring'
        run_name = f'roaring_f{int(shock_frac*100):02d}_seed{SEED}_finetuned'

    out_dir = os.path.join(REPO_ROOT, 'outputs', 'models', run_name)
    history_path = os.path.join(out_dir, 'history.json')

    if os.path.exists(history_path):
        print(f"  [skip] {run_name} already trained")
    else:
        print(f"  → training {run_name}…")
        env = os.environ.copy()
        env['PYENV_VERSION'] = '3.10.14'
        subprocess.run([
            'python', 'training/finetune.py',
            '--sampler', sampler,
            '--seed', str(SEED),
            '--shock-frac', str(shock_frac),
            '--epochs', str(EPOCHS),
            '--output-dir', run_name,
        ], cwd=REPO_ROOT, env=env, check=True)

    backtest_json = os.path.join(out_dir, 'backtest.json')
    if not os.path.exists(backtest_json):
        print(f"  → backtest {run_name}…")
        env = os.environ.copy()
        env['PYENV_VERSION'] = '3.10.14'
        subprocess.run([
            'python', 'eval/run_backtest.py',
            '--checkpoint', os.path.join(out_dir, 'best_model'),
            '--output', backtest_json,
            '--label', run_name,
        ], cwd=REPO_ROOT, env=env, check=True)

    with open(backtest_json) as f:
        bt = json.load(f)
    with open(history_path) as f:
        hist = json.load(f)
    return {
        'rank_ic':       bt['sig_last']['rank_ic'],
        'cum_return':    bt['sig_last']['total_return'],
        'sharpe':        bt['sig_last']['sharpe'],
        'max_drawdown':  bt['sig_last']['max_drawdown'],
        'val_loss':      hist['best_val_loss'],
    }


def main():
    print("=" * 66)
    print(f"shock_frac sweep — {len(SHOCK_FRACS)} values × 1 seed")
    print("=" * 66)

    results = {}
    for f in SHOCK_FRACS:
        print(f"\n── shock_frac={f:.2f} ─────────────")
        try:
            results[f] = run_one(f)
        except Exception as e:
            print(f"  ✗ failed: {e}")
            continue
        m = results[f]
        print(f"  IC={m['rank_ic']:+.4f}  CumRet={m['cum_return']:+.2%}  "
              f"Sharpe={m['sharpe']:+.3f}  MaxDD={m['max_drawdown']:+.2%}  "
              f"val_loss={m['val_loss']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    out_json = os.path.join(RESULTS_DIR, 'shock_frac_sweep.json')
    with open(out_json, 'w') as f:
        json.dump({
            'config': {'shock_fracs': SHOCK_FRACS, 'seed': SEED, 'epochs': EPOCHS},
            'results': {str(k): v for k, v in results.items()},
        }, f, indent=2)
    print(f"\n✓ Saved {out_json}")

    # ── Plot sensitivity curves ──────────────────────────────────────
    if not results:
        return

    fs = sorted(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("shock_frac sensitivity — backtest metrics by anchoring fraction",
                 fontsize=12, fontweight='bold')

    plots = [
        (axes[0, 0], 'rank_ic',      'RankIC',                 True),
        (axes[0, 1], 'cum_return',   'Cumulative return',      True),
        (axes[1, 0], 'sharpe',       'Sharpe ratio',           True),
        (axes[1, 1], 'max_drawdown', 'Max drawdown',          False),
    ]
    for ax, k, title, higher_better in plots:
        vals = [results[f][k] for f in fs]
        ax.plot(fs, vals, 'o-', markersize=8, linewidth=2, color='coral')
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)

        # Mark best value
        best_idx = int(np.argmax(vals)) if higher_better else int(np.argmin(vals))
        ax.scatter([fs[best_idx]], [vals[best_idx]], s=200, marker='*',
                   color='gold', edgecolor='black', zorder=5,
                   label=f'best: f={fs[best_idx]:.2f}')

        ax.set_xlabel('shock_frac')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight')
    print(f"✓ Saved {PLOT_PATH}")

    # ── Print paper-ready table ──────────────────────────────────────
    print("\n" + "=" * 66)
    print("PAPER-READY SENSITIVITY TABLE")
    print("=" * 66)
    print(f"{'shock_frac':<12}{'RankIC':>10}{'CumRet':>10}{'Sharpe':>10}{'MaxDD':>10}{'ValLoss':>10}")
    print("-" * 62)
    for f in fs:
        m = results[f]
        print(f"{f:<12.2f}{m['rank_ic']:>+10.4f}{m['cum_return']:>+9.2%}"
              f"{m['sharpe']:>+10.3f}{m['max_drawdown']:>+9.2%}"
              f"{m['val_loss']:>10.4f}")


if __name__ == '__main__':
    main()
