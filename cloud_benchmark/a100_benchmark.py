"""
Cloud A100 benchmark — measure the *real* data-loading fraction.

Runs both vanilla and Roaring dataloaders against the actual Kronos predictor
on whatever GPU is available. Measures per-step time breakdown:

  - data_loading_ms  (CPU work to assemble batch)
  - h2d_transfer_ms  (CPU → GPU memory copy)
  - forward_ms       (model forward pass)
  - backward_ms      (loss + backprop)
  - step_ms          (optimizer step + zero_grad)
  - total_ms

Output: results.json + bench.png with the full breakdown.

This is the script to run on Lambda/RunPod A100 to lock in the real
data-loading fraction (currently estimated at 25% in our projection).

Self-contained: clones the repo, downloads data, runs both benchmarks,
saves results. Expected runtime: ~3-5 minutes on A100.

Run on cloud GPU (after setup.sh):
    python a100_benchmark.py --model NeoQuasar/Kronos-base --batches 100
"""

import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kronos_src'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import KronosTokenizer, Kronos
from training.roaring_dataloader import VanillaDataset, RoaringDataset


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_loader(loader, model, tokenizer, optimizer, device, n_batches, label):
    """Run n_batches end-to-end and time each phase."""
    timings = {
        'data_loading_ms': [],
        'h2d_transfer_ms': [],
        'tokenize_ms':     [],
        'forward_ms':      [],
        'backward_ms':     [],
        'step_ms':         [],
    }

    model.train()
    loader_iter = iter(loader)

    # Warmup (don't count first 5 batches)
    for _ in range(5):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        x, stamp, s1_ids, s2_ids, _ = batch
        x = x.to(device); stamp = stamp.to(device)
        with torch.no_grad():
            indices = tokenizer.encode(x, half=True)
        s1_logits, s2_logits = model(indices[0][:, :-1], indices[1][:, :-1],
                                      stamp[:, :-1, :])
        loss, _, _ = model.head.compute_loss(
            s1_logits, s2_logits, indices[0][:, 1:], indices[1][:, 1:])
        loss.backward()
        optimizer.zero_grad()
    cuda_sync()

    print(f"  [{label}] warmup done, timing {n_batches} batches…")

    for i in range(n_batches):
        # — Phase 1: data loading (CPU)
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        x_cpu, stamp_cpu, s1_cpu, s2_cpu, _ = batch
        t_data = time.perf_counter()

        # — Phase 2: H2D transfer
        x = x_cpu.to(device, non_blocking=True)
        stamp = stamp_cpu.to(device, non_blocking=True)
        cuda_sync()
        t_h2d = time.perf_counter()

        # — Phase 3: tokenize on-the-fly (GPU)
        with torch.no_grad():
            indices = tokenizer.encode(x, half=True)
        cuda_sync()
        t_tok = time.perf_counter()

        # — Phase 4: forward
        s1_logits, s2_logits = model(indices[0][:, :-1], indices[1][:, :-1],
                                      stamp[:, :-1, :])
        loss, _, _ = model.head.compute_loss(
            s1_logits, s2_logits, indices[0][:, 1:], indices[1][:, 1:])
        cuda_sync()
        t_fwd = time.perf_counter()

        # — Phase 5: backward
        loss.backward()
        cuda_sync()
        t_bwd = time.perf_counter()

        # — Phase 6: optimizer step
        optimizer.step()
        optimizer.zero_grad()
        cuda_sync()
        t_step = time.perf_counter()

        timings['data_loading_ms'].append((t_data - t0)    * 1000)
        timings['h2d_transfer_ms'].append((t_h2d - t_data) * 1000)
        timings['tokenize_ms'].append((t_tok - t_h2d)      * 1000)
        timings['forward_ms'].append((t_fwd - t_tok)       * 1000)
        timings['backward_ms'].append((t_bwd - t_fwd)      * 1000)
        timings['step_ms'].append((t_step - t_bwd)         * 1000)

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_batches} batches done")

    # Aggregate
    summary = {phase: {
        'mean_ms': float(np.mean(vals)),
        'median_ms': float(np.median(vals)),
        'std_ms': float(np.std(vals)),
        'p95_ms': float(np.percentile(vals, 95)),
    } for phase, vals in timings.items()}

    mean_total = sum(s['mean_ms'] for s in summary.values())
    for phase in summary:
        summary[phase]['fraction'] = summary[phase]['mean_ms'] / mean_total
    summary['_total'] = {
        'mean_ms': mean_total,
        'data_fraction': (summary['data_loading_ms']['mean_ms'] +
                          summary['h2d_transfer_ms']['mean_ms']) / mean_total,
        'compute_fraction': (summary['tokenize_ms']['mean_ms'] +
                             summary['forward_ms']['mean_ms'] +
                             summary['backward_ms']['mean_ms'] +
                             summary['step_ms']['mean_ms']) / mean_total,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='NeoQuasar/Kronos-base',
                        help='HF model id (Kronos-small | Kronos-base | Kronos-large)')
    parser.add_argument('--batches', type=int, default=100,
                        help='Number of batches to time per sampler')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Per-GPU batch size')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Sequence length (Kronos max_context = 512)')
    parser.add_argument('--shock-frac', type=float, default=0.30)
    parser.add_argument('--output', default='results.json')
    args = parser.parse_args()

    print("="*60)
    print(f"Cloud A100 Benchmark")
    print(f"  Model:       {args.model}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Seq len:     {args.seq_len}")
    print(f"  Batches:     {args.batches}")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading {args.model}…")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device).eval()
    for p in tokenizer.parameters(): p.requires_grad_(False)
    model = Kronos.from_pretrained(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.1f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)

    print(f"\nBuilding datasets (seq_len={args.seq_len})…")
    v_ds = VanillaDataset(seq_len=args.seq_len,
                          steps_per_epoch=(args.batches + 10) * args.batch_size, seed=1)
    r_ds = RoaringDataset(seq_len=args.seq_len,
                          steps_per_epoch=(args.batches + 10) * args.batch_size, seed=1,
                          rare_threshold=5, shock_frac=args.shock_frac)
    v_loader = DataLoader(v_ds, batch_size=args.batch_size,
                          num_workers=2, pin_memory=True)
    r_loader = DataLoader(r_ds, batch_size=args.batch_size,
                          num_workers=2, pin_memory=True)

    print("\n" + "="*60)
    print("VANILLA")
    print("="*60)
    vanilla_summary = time_loader(v_loader, model, tokenizer, optimizer,
                                   device, args.batches, "vanilla")

    print("\n" + "="*60)
    print("ROARING")
    print("="*60)
    roaring_summary = time_loader(r_loader, model, tokenizer, optimizer,
                                   device, args.batches, "roaring")

    # ── Print summary ──
    print("\n" + "="*70)
    print("PER-STEP TIME BREAKDOWN (mean ms per batch)")
    print("="*70)
    phases = ['data_loading_ms', 'h2d_transfer_ms', 'tokenize_ms',
              'forward_ms', 'backward_ms', 'step_ms']
    print(f"  {'Phase':<22} {'Vanilla':>12} {'Roaring':>12} {'Δ':>10}")
    print("  " + "-"*60)
    for p in phases:
        v = vanilla_summary[p]['mean_ms']
        r = roaring_summary[p]['mean_ms']
        delta = r - v
        print(f"  {p:<22} {v:>10.2f}ms {r:>10.2f}ms {delta:>+8.2f}ms")
    print("  " + "-"*60)
    vt = vanilla_summary['_total']['mean_ms']
    rt = roaring_summary['_total']['mean_ms']
    print(f"  {'TOTAL':<22} {vt:>10.2f}ms {rt:>10.2f}ms {rt-vt:>+8.2f}ms")
    print()
    print(f"  Vanilla data fraction: {vanilla_summary['_total']['data_fraction']:.1%}")
    print(f"  Roaring data fraction: {roaring_summary['_total']['data_fraction']:.1%}")
    print(f"  End-to-end speedup:    {vt/rt:.3f}×")

    # ── Save ──
    out = {
        'config': vars(args),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'model_params_M': n_params / 1e6,
        'vanilla': vanilla_summary,
        'roaring': roaring_summary,
        'end_to_end_speedup': vt / rt,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {args.output}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        v_vals = [vanilla_summary[p]['mean_ms'] for p in phases]
        r_vals = [roaring_summary[p]['mean_ms'] for p in phases]
        phase_labels = [p.replace('_ms', '').replace('_', ' ') for p in phases]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"A100 Step-Time Breakdown — {args.model} (B={args.batch_size}, T={args.seq_len})",
                     fontsize=12, fontweight='bold')

        # Stacked bar
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#27ae60', '#3498db', '#9b59b6']
        bottom_v = bottom_r = 0
        for i, p in enumerate(phases):
            ax1.bar(0, v_vals[i], bottom=bottom_v, color=colors[i],
                    label=phase_labels[i], edgecolor='white', linewidth=0.5)
            ax1.bar(1, r_vals[i], bottom=bottom_r, color=colors[i],
                    edgecolor='white', linewidth=0.5)
            bottom_v += v_vals[i]; bottom_r += r_vals[i]
        ax1.set_xticks([0, 1]); ax1.set_xticklabels(['Vanilla', 'Roaring'])
        ax1.set_ylabel('Mean ms / batch')
        ax1.set_title('Stacked step-time breakdown')
        ax1.legend(loc='upper right', fontsize=9)

        # Side-by-side
        x = np.arange(len(phases))
        w = 0.35
        ax2.bar(x - w/2, v_vals, w, label='Vanilla', color='steelblue', alpha=0.85)
        ax2.bar(x + w/2, r_vals, w, label='Roaring', color='coral',     alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(phase_labels, rotation=20)
        ax2.set_ylabel('Mean ms / batch')
        ax2.set_title('Per-phase comparison')
        ax2.legend(); ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('bench.png', dpi=140, bbox_inches='tight')
        print("Plot saved → bench.png")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == '__main__':
    main()
