"""
Epoch-count sweep — addresses the 'short fine-tune' limitation.

Trains both samplers for 5, 10, 20, 30 epochs each and reports the
val-loss + backtest metrics at each checkpoint. Determines whether:
  - Roaring's slight aggregate disadvantage at 5 epochs persists,
    closes, or flips at longer training
  - Either sampler overfits (val loss starts increasing)
  - The drawdown and per-month-IC patterns persist or wash out

Output:
  outputs/models/<sampler>_e<E>_seed42_finetuned/  — 8 checkpoint dirs
  experiments/results/epoch_sweep.json              — aggregated metrics
  experiments/epoch_sweep.png                       — convergence plot

Paper deliverable: new figure showing val-loss and backtest-metric
trajectories vs epoch count. Likely replaces or augments the
limitation 'Short fine-tune (5 epochs)'.

Run (~3 hours on MacBook MPS):
  PYENV_VERSION=3.10.14 python experiments/epoch_sweep.py
"""

import sys, os, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, 'experiments', 'results')
PLOT_PATH   = os.path.join(REPO_ROOT, 'experiments', 'epoch_sweep.png')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCH_VALUES = [5, 10, 20, 30]
SAMPLERS     = ['vanilla', 'roaring']
SEED         = 42


def run_one(sampler: str, epochs: int) -> dict:
    run_name = f'{sampler}_e{epochs:02d}_seed{SEED}_finetuned'
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
            '--epochs', str(epochs),
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

    with open(backtest_json) as f: bt = json.load(f)
    with open(history_path)  as f: hist = json.load(f)
    return {
        'val_loss_per_epoch': [h['val_loss'] for h in hist['history']],
        'train_loss_per_epoch': [h['train_loss'] for h in hist['history']],
        'best_val_loss': hist['best_val_loss'],
        'rank_ic':      bt['sig_last']['rank_ic'],
        'cum_return':   bt['sig_last']['total_return'],
        'sharpe':       bt['sig_last']['sharpe'],
        'max_drawdown': bt['sig_last']['max_drawdown'],
    }


def main():
    print("=" * 66)
    print(f"Epoch sweep — {len(EPOCH_VALUES)} values × {len(SAMPLERS)} samplers")
    print("=" * 66)

    results = {s: {} for s in SAMPLERS}
    for sampler in SAMPLERS:
        for e in EPOCH_VALUES:
            print(f"\n── {sampler} e={e} ─────────────")
            try:
                results[sampler][e] = run_one(sampler, e)
                m = results[sampler][e]
                print(f"  best_val={m['best_val_loss']:.4f}  "
                      f"IC={m['rank_ic']:+.4f}  Sharpe={m['sharpe']:+.3f}  "
                      f"MaxDD={m['max_drawdown']:+.2%}")
            except Exception as e_:
                print(f"  ✗ failed: {e_}")

    # ── Save ─────────────────────────────────────────────────────────
    out_json = os.path.join(RESULTS_DIR, 'epoch_sweep.json')
    with open(out_json, 'w') as f:
        json.dump({
            'config': {'epoch_values': EPOCH_VALUES, 'samplers': SAMPLERS, 'seed': SEED},
            'results': {s: {str(e): m for e, m in r.items()} for s, r in results.items()},
        }, f, indent=2)
    print(f"\n✓ Saved {out_json}")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Epoch-count sweep: vanilla vs Roaring at varying training durations",
                 fontsize=12, fontweight='bold')

    # Validation loss trajectories
    ax = axes[0, 0]
    longest = max(EPOCH_VALUES)
    for sampler, color in zip(SAMPLERS, ['steelblue', 'coral']):
        if longest in results[sampler]:
            losses = results[sampler][longest]['val_loss_per_epoch']
            ax.plot(range(1, len(losses)+1), losses, 'o-', color=color,
                    label=sampler, linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val loss')
    ax.set_title(f'Val loss curves ({longest}-epoch run)')
    ax.legend(); ax.grid(alpha=0.3)

    # Best val loss vs epoch count
    ax = axes[0, 1]
    for sampler, color in zip(SAMPLERS, ['steelblue', 'coral']):
        epochs = sorted(results[sampler].keys())
        if epochs:
            vals = [results[sampler][e]['best_val_loss'] for e in epochs]
            ax.plot(epochs, vals, 'o-', color=color, label=sampler,
                    linewidth=2, markersize=8)
    ax.set_xlabel('Epoch count'); ax.set_ylabel('Best val loss')
    ax.set_title('Best validation loss vs training length')
    ax.legend(); ax.grid(alpha=0.3)

    # RankIC vs epoch count
    ax = axes[1, 0]
    for sampler, color in zip(SAMPLERS, ['steelblue', 'coral']):
        epochs = sorted(results[sampler].keys())
        if epochs:
            vals = [results[sampler][e]['rank_ic'] for e in epochs]
            ax.plot(epochs, vals, 'o-', color=color, label=sampler,
                    linewidth=2, markersize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Epoch count'); ax.set_ylabel('RankIC (sig_last)')
    ax.set_title('Backtest RankIC vs training length')
    ax.legend(); ax.grid(alpha=0.3)

    # Max drawdown vs epoch count
    ax = axes[1, 1]
    for sampler, color in zip(SAMPLERS, ['steelblue', 'coral']):
        epochs = sorted(results[sampler].keys())
        if epochs:
            vals = [results[sampler][e]['max_drawdown'] * 100 for e in epochs]
            ax.plot(epochs, vals, 'o-', color=color, label=sampler,
                    linewidth=2, markersize=8)
    ax.set_xlabel('Epoch count'); ax.set_ylabel('Max drawdown (%)')
    ax.set_title('Drawdown vs training length')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight')
    print(f"✓ Saved {PLOT_PATH}")

    # ── Print paper-ready table ──────────────────────────────────────
    print("\n" + "=" * 66)
    print("EPOCH-SWEEP RESULTS TABLE")
    print("=" * 66)
    print(f"{'Epochs':<8} ", end="")
    for s in SAMPLERS:
        print(f"  {s+' val_loss':>16}", end="")
    for s in SAMPLERS:
        print(f"  {s+' RankIC':>14}", end="")
    print()
    print("-" * 90)
    for e in EPOCH_VALUES:
        print(f"{e:<8} ", end="")
        for s in SAMPLERS:
            v = results[s].get(e, {}).get('best_val_loss', float('nan'))
            print(f"  {v:>16.4f}", end="")
        for s in SAMPLERS:
            v = results[s].get(e, {}).get('rank_ic', float('nan'))
            print(f"  {v:>+14.4f}", end="")
        print()


if __name__ == '__main__':
    main()
