"""
train_audio.py — Antreneaza Audio Transformer pe MFCC + Delta + DeltaDelta
Dataset: CREMA-D + RAVDESS + SAVEE + TESS (4 dataset-uri combinate)
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import *
from models import AudioTransformer
from datasets import AudioMFCCDataset, EMOTION_TO_IDX
from train_utils import (
    FocalLoss, WarmupCosineScheduler,
    evaluate_predictions, print_metrics,
)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    bar = tqdm(loader, desc="  Train", leave=False)
    for features, labels in bar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * features.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[AUDIO] Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Dataloaders ──────────────────────────────────────────────────────
    train_csv = PROJECT_DIR / "audio_train.csv"
    val_csv   = PROJECT_DIR / "audio_val.csv"
    test_csv  = PROJECT_DIR / "audio_test.csv"

    for f in [train_csv, val_csv, test_csv]:
        if not f.exists():
            print(f"EROARE: {f} nu exista. Ruleaza mai intai: python prepare_data.py")
            sys.exit(1)

    # Determina dimensiunile
    n_samples = int(AUDIO_SR * AUDIO_DURATION)
    max_len = 1 + n_samples // AUDIO_HOP_LENGTH
    feature_dim = AUDIO_N_MFCC * 3  # MFCC + Delta + DeltaDelta = 40 * 3 = 120

    print(f"  Audio config: sr={AUDIO_SR}, duration={AUDIO_DURATION}s")
    print(f"  Feature dim: {feature_dim} (MFCC={AUDIO_N_MFCC} × 3)")
    print(f"  Sequence length: {max_len} frames")

    train_ds = AudioMFCCDataset(train_csv, max_len=max_len, augment=True)
    val_ds   = AudioMFCCDataset(val_csv,   max_len=max_len, augment=False)
    test_ds  = AudioMFCCDataset(test_csv,  max_len=max_len, augment=False)

    train_dl = DataLoader(train_ds, batch_size=AUDIO_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=AUDIO_BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=AUDIO_BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ────────────────────────────────────────────────────────────
    model = AudioTransformer(
        input_dim=feature_dim,
        num_classes=NUM_CLASSES,
        d_model=AUDIO_D_MODEL,
        nhead=AUDIO_NHEAD,
        num_layers=AUDIO_NUM_LAYERS,
        dim_ff=AUDIO_DIM_FF,
        dropout=AUDIO_DROPOUT,
        embed_dim=AUDIO_EMBED_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Class weights
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    train_labels = train_df['emotion'].map(EMOTION_TO_IDX).dropna().astype(int).values
    cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f"  Class weights: {cw.round(3)}")

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1, weight=class_weights)
    val_criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=AUDIO_LR,
        weight_decay=AUDIO_WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * AUDIO_EPOCHS
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=steps_per_epoch * 5,  # 5 epoci warmup
        total_steps=total_steps,
    )

    # ── Antrenare ────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_path = MODELS_DIR / "audio_transformer_best.pt"
    patience = 12
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"ANTRENARE AUDIO TRANSFORMER ({AUDIO_EPOCHS} epoci)")
    print(f"  d_model={AUDIO_D_MODEL}, heads={AUDIO_NHEAD}, "
          f"layers={AUDIO_NUM_LAYERS}, ff={AUDIO_DIM_FF}")
    print(f"{'='*60}")

    for epoch in range(1, AUDIO_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_dl, optimizer, scheduler, criterion, device
        )
        vl_loss, vl_acc, preds, labels = evaluate(
            model, val_dl, val_criterion, device
        )
        dt = time.time() - t0

        print(f"Ep {epoch:02d}/{AUDIO_EPOCHS} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"LR: {scheduler.get_lr()[0]:.2e} | Time: {dt:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': vl_acc,
                'class_names': EMOTION_CLASSES,
                'audio_config': {
                    'input_dim': feature_dim,
                    'd_model': AUDIO_D_MODEL,
                    'nhead': AUDIO_NHEAD,
                    'num_layers': AUDIO_NUM_LAYERS,
                    'dim_ff': AUDIO_DIM_FF,
                    'max_len': max_len,
                },
            }, best_path)
            print(f"  ✓ Best saved (val_acc={vl_acc:.4f})")
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == AUDIO_EPOCHS:
            metrics = evaluate_predictions(labels, preds)
            print(metrics['report'])

        if no_improve >= patience:
            print(f"\nEarly stopping la epoca {epoch} (patience={patience})")
            break

    # ══════════════════════════════════════════════════════════════════════
    # EVALUARE FINALA PE TEST SET
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("EVALUARE FINALA PE TEST SET")
    print(f"{'='*60}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Model incarcat din epoca {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    _, test_acc, test_preds, test_labels = evaluate(
        model, test_dl, val_criterion, device
    )
    metrics = evaluate_predictions(test_labels, test_preds)
    print_metrics(metrics, "AUDIO TRANSFORMER — Test Set Results")

    np.save(RESULTS_DIR / "audio_test_preds.npy", np.array(test_preds))
    np.save(RESULTS_DIR / "audio_test_labels.npy", np.array(test_labels))

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model salvat: {best_path}")
    print(f"Training complete!")


if __name__ == '__main__':
    main()
