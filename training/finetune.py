"""
Fine-tune Kronos-small on BTC 1h data.

Mirrors Kronos's official train_predictor.py training loop exactly:
  - Tokenize on-the-fly with the frozen tokenizer
  - Autoregressive next-token prediction (cross-entropy on s1 + s2)
  - AdamW + OneCycleLR

Two modes controlled by --sampler flag:
  vanilla  → uniform random window sampling (baseline)
  roaring  → Roaring Bitmap stratified sampling (30% shock bars)

Checkpoints saved to outputs/models/<sampler>_finetuned/

Run:
  PYENV_VERSION=3.10.14 python training/finetune.py --sampler vanilla
  PYENV_VERSION=3.10.14 python training/finetune.py --sampler roaring
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kronos_src'))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import KronosTokenizer, Kronos
from training.roaring_dataloader import VanillaDataset, RoaringDataset

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN          = 128      # bars per training window
BATCH_SIZE       = 16       # fits comfortably in MacBook RAM
EPOCHS           = 5
STEPS_PER_EPOCH  = 200      # batches per epoch (epoch = one pass over sampler)
VAL_STEPS        = 40       # batches per validation
LR               = 4e-5     # matches Kronos config.py predictor_learning_rate
WEIGHT_DECAY     = 0.1
BETA1, BETA2     = 0.9, 0.95
GRAD_CLIP        = 3.0
SHOCK_FRAC       = 0.3      # Roaring: 30% of windows anchored at shock bars
RARE_THRESHOLD   = 5        # tokens appearing ≤ 5 times = shock
LOG_EVERY        = 20       # print loss every N batches
SAVE_DIR         = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')


def make_dataset(sampler: str, split: str, seed: int, shock_frac: float = SHOCK_FRAC):
    """Build train or val dataset for the given sampler type."""
    # Validation uses vanilla sampling regardless of sampler type
    # so both models see the same val distribution
    steps = STEPS_PER_EPOCH if split == 'train' else VAL_STEPS
    effective_seed = seed if split == 'train' else 9999

    if sampler == 'roaring' and split == 'train':
        return RoaringDataset(
            seq_len=SEQ_LEN,
            steps_per_epoch=steps * BATCH_SIZE,
            seed=effective_seed,
            split=split,
            rare_threshold=RARE_THRESHOLD,
            shock_frac=shock_frac,
        )
    else:
        # Validation always uses uniform sampling over its own held-out region,
        # so both samplers are scored on the same leakage-free val distribution.
        return VanillaDataset(
            seq_len=SEQ_LEN,
            steps_per_epoch=steps * BATCH_SIZE,
            seed=effective_seed,
            split=split,
        )


def run_epoch(model, tokenizer, loader, optimizer, scheduler, device,
              is_train: bool, epoch: int):
    model.train(is_train)
    total_loss = total_s1 = total_s2 = 0.0
    n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, (x, stamp, s1_ids, s2_ids, _) in enumerate(loader):
            x      = x.to(device)        # (B, T, 6)
            stamp  = stamp.to(device)    # (B, T, 5)
            s1_ids = s1_ids.to(device)   # (B, T)
            s2_ids = s2_ids.to(device)   # (B, T)

            # Tokenize on-the-fly with frozen tokenizer (matches official training)
            with torch.no_grad():
                indices = tokenizer.encode(x, half=True)  # [s1(B,T), s2(B,T)]
                tok_s1 = indices[0]   # (B, T)
                tok_s2 = indices[1]   # (B, T)

            # Autoregressive: input = [:-1], target = [1:]
            inp_s1  = tok_s1[:, :-1]        # (B, T-1)
            inp_s2  = tok_s2[:, :-1]
            tgt_s1  = tok_s1[:, 1:]         # (B, T-1)
            tgt_s2  = tok_s2[:, 1:]
            inp_stamp = stamp[:, :-1, :]    # (B, T-1, 5)

            # Forward pass
            s1_logits, s2_logits = model(inp_s1, inp_s2, inp_stamp)

            # Loss (cross-entropy on both heads, matches Kronos exactly)
            loss, s1_loss, s2_loss = model.head.compute_loss(
                s1_logits, s2_logits, tgt_s1, tgt_s2
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            total_s1   += s1_loss.item()
            total_s2   += s2_loss.item()
            n += 1

            if is_train and (step + 1) % LOG_EVERY == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"    step {step+1}/{STEPS_PER_EPOCH}  "
                      f"loss={loss.item():.4f}  "
                      f"s1={s1_loss.item():.4f}  s2={s2_loss.item():.4f}  "
                      f"lr={lr:.2e}")

    return total_loss / n, total_s1 / n, total_s2 / n


def main(sampler: str, seed: int = 42, shock_frac: float = SHOCK_FRAC,
         epochs: int = EPOCHS, output_dir: str | None = None):
    print(f"\n{'='*60}")
    print(f"Fine-tuning Kronos-small  |  sampler={sampler}  seed={seed}  "
          f"shock_frac={shock_frac}  epochs={epochs}")
    print(f"{'='*60}\n")

    # Seed everything for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}\n")

    # ── Load pretrained models ─────────────────────────────────────────────
    print("Loading Kronos-Tokenizer-base (frozen)…")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    tokenizer = tokenizer.to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    print("Loading Kronos-small (to be fine-tuned)…")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

    # ── Dataloaders ────────────────────────────────────────────────────────
    print("Building datasets…")
    train_ds = make_dataset(sampler, 'train', seed=seed, shock_frac=shock_frac)
    val_ds   = make_dataset(sampler, 'val',   seed=seed, shock_frac=shock_frac)
    print()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR,
        betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
    )
    total_steps = STEPS_PER_EPOCH * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        total_steps=total_steps,
        pct_start=0.05, div_factor=10, final_div_factor=100
    )

    # ── Save dir ──────────────────────────────────────────────────────────
    run_name = output_dir or f'{sampler}_finetuned'
    save_path = os.path.join(SAVE_DIR, run_name, 'best_model')
    os.makedirs(save_path, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        print(f"── Epoch {epoch}/{epochs} ──────────────────────────────────")

        train_loss, train_s1, train_s2 = run_epoch(
            model, tokenizer, train_loader, optimizer, scheduler,
            device, is_train=True, epoch=epoch
        )
        val_loss, val_s1, val_s2 = run_epoch(
            model, tokenizer, val_loader, None, None,
            device, is_train=False, epoch=epoch
        )

        elapsed = time.perf_counter() - t0
        print(f"\n  train_loss={train_loss:.4f}  (s1={train_s1:.4f}, s2={train_s2:.4f})")
        print(f"  val_loss  ={val_loss:.4f}  (s1={val_s1:.4f}, s2={val_s2:.4f})")
        print(f"  time: {elapsed:.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_s1': train_s1, 'train_s2': train_s2,
            'val_loss':   val_loss,   'val_s1':   val_s1,   'val_s2':   val_s2,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(save_path)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"Training complete  |  sampler={sampler}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {save_path}")
    print(f"{'='*60}\n")

    import json
    with open(os.path.join(SAVE_DIR, run_name, 'history.json'), 'w') as f:
        json.dump({
            'config': {
                'sampler': sampler, 'seed': seed, 'shock_frac': shock_frac,
                'epochs': epochs, 'batch_size': BATCH_SIZE, 'seq_len': SEQ_LEN,
                'lr': LR, 'steps_per_epoch': STEPS_PER_EPOCH,
            },
            'history': history,
            'best_val_loss': best_val_loss,
        }, f, indent=2)
    print("Training history saved.")
    return best_val_loss, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--shock-frac', type=float, default=SHOCK_FRAC,
                        help='Fraction of windows anchored at shock bars (Roaring only)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory name under outputs/models/. '
                             'Defaults to "<sampler>_finetuned"')
    parser.add_argument('--sampler', choices=['vanilla', 'roaring'], default='vanilla',
                        help='vanilla = uniform sampling | roaring = stratified shock sampling')
    args = parser.parse_args()
    main(args.sampler, seed=args.seed, shock_frac=args.shock_frac,
         epochs=args.epochs, output_dir=args.output_dir)
