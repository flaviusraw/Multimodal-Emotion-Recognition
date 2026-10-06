"""
train_video_frozen_only.py — Antreneaza ViViT DOAR cu backbone frozen (Faza 1)
Folosit pentru a evalua performanta modelului fara fine-tuning.

Bazat pe train_video_haar.py — Faza 2 (fine-tuning) dezactivata.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import *
from models import VideoViViT
from datasets import VideoHaarDataset
from config import PROJECT_DIR

# Director cu frame-urile JPEG pre-extrase
HAAR_FRAMES_DIR = PROJECT_DIR / "haar_frames"

from train_utils import (
    FocalLoss, WarmupCosineScheduler,
    evaluate_predictions, print_metrics,
)


def train_one_epoch_warmup(model, loader, optimizer, scheduler, criterion,
                           device, scaler):
    """Faza 1: backbone frozen, fara MixUp, cu AMP."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    bar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[VIDEO - FROZEN ONLY] Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Dataloaders ──────────────────────────────────────────────────────
    train_csv = PROJECT_DIR / "train.csv"
    val_csv   = PROJECT_DIR / "val.csv"
    test_csv  = PROJECT_DIR / "test.csv"

    for f in [train_csv, val_csv, test_csv]:
        if not f.exists():
            print(f"EROARE: {f} nu exista. Ruleaza mai intai: python prepare_data.py")
            sys.exit(1)

    train_ds = VideoHaarDataset(train_csv, HAAR_FRAMES_DIR, augment=True)
    val_ds   = VideoHaarDataset(val_csv,   HAAR_FRAMES_DIR)
    test_ds  = VideoHaarDataset(test_csv,  HAAR_FRAMES_DIR)

    train_dl = DataLoader(train_ds, batch_size=VIDEO_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=VIDEO_BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=VIDEO_BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Batch size: {VIDEO_BATCH_SIZE}")

    # ── Model ────────────────────────────────────────────────────────────
    model = VideoViViT(num_classes=NUM_CLASSES, embed_dim=VIDEO_EMBED_DIM,
                       dropout=VIDEO_DROPOUT).to(device)
    total_params  = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # ── Class weights ────────────────────────────────────────────────────
    train_df = pd.read_csv(train_csv)
    train_labels = train_df['emotion'].map(
        {e: i for i, e in enumerate(EMOTION_CLASSES)}
    ).dropna().astype(int).values
    cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f"  Class weights: {cw.round(3)}")

    criterion     = FocalLoss(gamma=2.0, label_smoothing=0.1, weight=class_weights)
    val_criterion = nn.CrossEntropyLoss()

    scaler   = GradScaler("cuda", enabled=(device == "cuda"))
    best_val_acc = 0.0
    best_path    = MODELS_DIR / f"video_vit_haar_frozen_best_{NUM_CLASSES}cls.pt"

    # ══════════════════════════════════════════════════════════════════════
    # FAZA 1: Warm-up — backbone frozen, doar head-ul se antreneaza
    # ══════════════════════════════════════════════════════════════════════
    FROZEN_EPOCHS = 25

    print(f"\n{'='*60}")
    print(f"FAZA 1 (SINGURA): Backbone frozen ({FROZEN_EPOCHS} epoci)")
    print(f"{'='*60}")

    model.freeze_backbone()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametri antrenabili (doar head): {trainable_params:,}")

    trainable  = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(trainable, lr=VIDEO_LR_HEAD,
                                   weight_decay=VIDEO_WEIGHT_DECAY)
    steps_warm = len(train_dl)
    scheduler  = WarmupCosineScheduler(optimizer, warmup_steps=steps_warm,
                                       total_steps=steps_warm * FROZEN_EPOCHS)

    patience   = 8
    no_improve = 0

    for epoch in range(1, FROZEN_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch_warmup(
            model, train_dl, optimizer, scheduler, criterion, device, scaler
        )
        vl_loss, vl_acc, preds, labels = evaluate(model, val_dl, val_criterion, device)
        dt = time.time() - t0

        print(f"Ep {epoch:02d}/{FROZEN_EPOCHS} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"Time: {dt:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save({
                'epoch': epoch,
                'phase': 'frozen_only',
                'model_state_dict': model.state_dict(),
                'val_acc': vl_acc,
                'class_names': EMOTION_CLASSES,
            }, best_path)
            print(f"  ✓ Best saved (val_acc={vl_acc:.4f})")
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == FROZEN_EPOCHS:
            metrics = evaluate_predictions(labels, preds)
            print(metrics['report'])

        if no_improve >= patience:
            print(f"\nEarly stopping la epoca {epoch} (patience={patience})")
            break

    # ══════════════════════════════════════════════════════════════════════
    # EVALUARE FINALA PE TEST SET
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("EVALUARE FINALA PE TEST SET — BACKBONE FROZEN")
    print(f"{'='*60}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Model incarcat din epoca {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    _, test_acc, test_preds, test_labels = evaluate(model, test_dl, val_criterion, device)
    metrics = evaluate_predictions(test_labels, test_preds)
    print_metrics(metrics, "VIDEO MODEL FROZEN — Test Set Results")

    np.save(RESULTS_DIR / "video_frozen_test_preds.npy",  np.array(test_preds))
    np.save(RESULTS_DIR / "video_frozen_test_labels.npy", np.array(test_labels))

    print(f"\nBest val accuracy  : {best_val_acc:.4f}")
    print(f"Test accuracy      : {test_acc:.4f}")
    print(f"Model salvat       : {best_path}")
    print(f"\nComparatie rapida:")
    print(f"  Frozen only  → val: {best_val_acc:.4f} | test: {test_acc:.4f}")
    print(f"  Full finetune→ val: 0.8352        | test: 0.8309  (referinta)")
    print(f"\nTraining complete!")


if __name__ == '__main__':
    main()
