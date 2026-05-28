"""
Multi-seed runner — addresses the #1 reviewer concern about single-seed
results.

For each of 3 seeds and 2 samplers (vanilla, roaring), runs the
full fine-tune + backtest pipeline and aggregates the headline
metrics (RankIC, CumRet, Sharpe, MaxDD) as mean ± std.

Output:
  outputs/models/<sampler>_seed<S>_finetuned/   — 6 checkpoint dirs
  experiments/results/multi_seed_summary.json   — aggregated metrics
  experiments/multi_seed_summary.png             — error-bar plot

Paper deliverable: replaces Table 4 single-point numbers with
mean ± std across seeds.

Run (~3-4 hours on MacBook MPS):
  PYENV_VERSION=3.10.14 python experiments/multi_seed_runner.py
"""

import sys, os, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, 'experiments', 'results')
PLOT_PATH   = os.path.join(REPO_ROOT, 'experiments', 'multi_seed_summary.png')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS    = [42, 137, 2026]
SAMPLERS = ['vanilla', 'roaring']
EPOCHS   = 5   # same as the single-seed paper run


def run_one(sampler: str, seed: int) -> dict:
    """Run one fine-tune + backtest combo. Returns metrics dict."""
    run_name = f'{sampler}_seed{seed}_finetuned'
    out_dir = os.path.join(REPO_ROOT, 'outputs', 'models', run_name)
    history_path = os.path.join(out_dir, 'history.json')

    # Skip if already done
    if os.path.exists(history_path):
        print(f"  [skip] {run_name} already complete")
    else:
        print(f"  → training {run_name}…")
        env = os.environ.copy()
        env['PYENV_VERSION'] = '3.10.14'
        subprocess.run([
            'python', 'training/finetune.py',
            '--sampler', sampler,
            '--seed', str(seed),
            '--epochs', str(EPOCHS),
            '--output-dir', run_name,
        ], cwd=REPO_ROOT, env=env, check=True)

    # Now run backtest with this checkpoint
    backtest_json = os.path.join(out_dir, 'backtest.json')
    if not os.path.exists(backtest_json):
        print(f"  → backtest {run_name}…")
        env = os.environ.copy()
        env['PYENV_VERSION'] = '3.10.14'
        subprocess.run([
            'python', 'eval/run_backtest.py',
            '--checkpoint', os.path.join(out_dir, 'best_model'),
            '--output', backtest_json,
        ], cwd=REPO_ROOT, env=env, check=True)

    with open(backtest_json) as f:
        bt = json.load(f)
    with open(history_path) as f:
        hist = json.load(f)
    return {'backtest': bt, 'history': hist}


def main():
    print("=" * 66)
    print(f"Multi-seed runner — {len(SEEDS)} seeds × {len(SAMPLERS)} samplers = {len(SEEDS)*len(SAMPLERS)} runs")
    print("=" * 66)

    all_runs = {}
    for sampler in SAMPLERS:
        all_runs[sampler] = []
        for seed in SEEDS:
            print(f"\n── {sampler} seed={seed} ─────────────")
            try:
                r = run_one(sampler, seed)
                all_runs[sampler].append({'seed': seed, **r})
            except Exception as e:
                print(f"  ✗ {sampler} seed={seed} failed: {e}")

    # ── Aggregate ─────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("AGGREGATED RESULTS (mean ± std across seeds)")
    print("=" * 66)

    summary = {}
    metrics_keys = ['rank_ic_last', 'cum_return', 'sharpe', 'max_drawdown', 'best_val_loss']

    for sampler in SAMPLERS:
        runs = all_runs[sampler]
        if not runs: continue

        # Flatten backtest metrics
        per_metric = {k: [] for k in metrics_keys}
        for r in runs:
            bt = r['backtest']
            per_metric['rank_ic_last'].append(bt['sig_last']['rank_ic'])
            per_metric['cum_return'].append(bt['sig_last']['total_return'])
            per_metric['sharpe'].append(bt['sig_last']['sharpe'])
            per_metric['max_drawdown'].append(bt['sig_last']['max_drawdown'])
            per_metric['best_val_loss'].append(r['history']['best_val_loss'])

        summary[sampler] = {}
        for k in metrics_keys:
            vals = np.array(per_metric[k])
            summary[sampler][k] = {
                'mean': float(vals.mean()),
                'std':  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                'n':    len(vals),
                'values': vals.tolist(),
            }

        print(f"\n{sampler:<10}", end="")
        for k in metrics_keys:
            s = summary[sampler][k]
            if 'loss' in k:
                print(f"  {k}={s['mean']:.4f}±{s['std']:.4f}", end="")
            elif 'ic' in k:
                print(f"  {k}={s['mean']:+.4f}±{s['std']:.4f}", end="")
            else:
                print(f"  {k}={s['mean']:+.2%}±{s['std']:.2%}", end="")
        print()

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, 'multi_seed_summary.json')
    with open(out_json, 'w') as f:
        json.dump({
            'config': {'seeds': SEEDS, 'samplers': SAMPLERS, 'epochs': EPOCHS},
            'summary': summary,
            'raw_runs': {s: [{'seed': r['seed'], **r['backtest']['sig_last'],
                              'best_val_loss': r['history']['best_val_loss']}
                             for r in all_runs[s]]
                         for s in SAMPLERS if all_runs[s]},
        }, f, indent=2)
    print(f"\n✓ Saved {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────
    if all(sampler in summary for sampler in SAMPLERS):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"Multi-seed comparison (n={len(SEEDS)} per sampler)",
                     fontsize=12, fontweight='bold')

        plot_metrics = [
            ('rank_ic_last',  'RankIC'),
            ('cum_return',    'Cumulative return'),
            ('sharpe',        'Sharpe ratio'),
            ('max_drawdown',  'Max drawdown'),
        ]
        for ax, (k, title) in zip(axes, plot_metrics):
            means = [summary[s][k]['mean'] for s in SAMPLERS]
            stds  = [summary[s][k]['std']  for s in SAMPLERS]
            colors = ['steelblue', 'coral']
            ax.bar(SAMPLERS, means, yerr=stds, capsize=8,
                   color=colors, alpha=0.85, edgecolor='black')
            ax.set_title(title); ax.grid(axis='y', alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.5)

            # raw points
            for i, s in enumerate(SAMPLERS):
                vals = summary[s][k]['values']
                ax.scatter([i]*len(vals), vals, color='black', s=20, zorder=3)

        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=140, bbox_inches='tight')
        print(f"✓ Saved {PLOT_PATH}")


if __name__ == '__main__':
    main()
