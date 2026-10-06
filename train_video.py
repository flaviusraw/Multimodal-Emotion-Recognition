"""
train_video.py — Antreneaza Vision Transformer pe imagini cu fete
Faza 1: Warm-up (backbone frozen, doar head-ul se antreneaza)
Faza 2: Fine-tuning (tot modelul, LR diferentiat)

Fix-uri aplicate:
  - CSV-uri unificate (train/val/test.csv) fara clasa 'sad'
  - model.vivit (nu model.backbone)
  - AMP (bfloat16) pentru reducere memorie ~50%
  - Gradient checkpointing pe ViViT encoder
  - Batch size 2 + grad_accum 4 in Faza 2 → effective batch 8
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import *
from models import VideoViViT
from datasets import VideoClipDataset
from train_utils import (
    FocalLoss, WarmupCosineScheduler,
    mixup_data, mixup_criterion,
    evaluate_predictions, print_metrics,
)

# ── Gradient accumulation pentru Faza 2 ────────────────────────────────────
# Batch 2 × acum 4 = effective batch 8 (ca in articol), folosind ~25% memorie
FINETUNE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS    = 4


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
        with autocast(dtype=torch.bfloat16):
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


def train_one_epoch_finetune(model, loader, optimizer, scheduler, criterion,
                             device, scaler, mixup_alpha=0.3):
    """Faza 2: tot modelul, MixUp, AMP, gradient accumulation."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    bar = tqdm(loader, desc="  Train", leave=False)
    optimizer.zero_grad()
    for step, (images, labels) in enumerate(bar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        mixed_images, mixed_labels = mixup_data(images, labels, mixup_alpha)

        with autocast(dtype=torch.bfloat16):
            logits = model(mixed_images)
            loss   = mixup_criterion(logits, mixed_labels) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * GRAD_ACCUM_STEPS * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
        bar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(dtype=torch.bfloat16):
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
    print(f"[VIDEO] Device: {device}")
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

    train_ds = VideoClipDataset(train_csv)
    val_ds   = VideoClipDataset(val_csv)
    test_ds  = VideoClipDataset(test_csv)

    # Faza 1 batch=8, Faza 2 batch=2 cu acumulare × 4 = effective 8
    train_dl_warm = DataLoader(train_ds, batch_size=VIDEO_BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    train_dl_fine = DataLoader(train_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dl  = DataLoader(val_ds,  batch_size=VIDEO_BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=VIDEO_BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Faza 1 batch: {VIDEO_BATCH_SIZE} | "
          f"Faza 2 batch: {FINETUNE_BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum "
          f"= {FINETUNE_BATCH_SIZE * GRAD_ACCUM_STEPS} effective")

    # ── Model ────────────────────────────────────────────────────────────
    model = VideoViViT(num_classes=NUM_CLASSES, embed_dim=VIDEO_EMBED_DIM,
                       dropout=VIDEO_DROPOUT).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Gradient checkpointing: salveaza ~40% memorie in Faza 2
    if hasattr(model.vivit, 'encoder'):
        model.vivit.encoder.gradient_checkpointing = True
        print("  Gradient checkpointing activat pe ViViT encoder")

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

    scaler = GradScaler(enabled=(device == 'cuda'))

    best_val_acc = 0.0
    best_path    = MODELS_DIR / f"video_vit_best_{NUM_CLASSES}cls.pt"

    # ══════════════════════════════════════════════════════════════════════
    # FAZA 1: Warm-up — backbone frozen
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"FAZA 1: Warm-up ({VIDEO_EPOCHS_WARM} epoci, backbone frozen)")
    print(f"{'='*60}")

    model.freeze_backbone()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=VIDEO_LR_HEAD,
                                  weight_decay=VIDEO_WEIGHT_DECAY)
    steps_warm = len(train_dl_warm)
    scheduler  = WarmupCosineScheduler(optimizer, warmup_steps=steps_warm,
                                       total_steps=steps_warm * VIDEO_EPOCHS_WARM)

    for epoch in range(1, VIDEO_EPOCHS_WARM + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch_warmup(
            model, train_dl_warm, optimizer, scheduler, criterion, device, scaler
        )
        vl_loss, vl_acc, _, _ = evaluate(model, val_dl, val_criterion, device)
        dt = time.time() - t0

        print(f"Ep {epoch:02d}/{VIDEO_EPOCHS_WARM} [warm] | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"Time: {dt:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                'epoch': epoch,
                'phase': 'warmup',
                'model_state_dict': model.state_dict(),
                'val_acc': vl_acc,
                'class_names': EMOTION_CLASSES,
            }, best_path)
            print(f"  Best saved (val_acc={vl_acc:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    # FAZA 2: Fine-tuning — backbone unfrozen, LR diferentiat
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"FAZA 2: Fine-tuning ({VIDEO_EPOCHS_FINE} epoci, tot modelul)")
    print(f"{'='*60}")

    torch.cuda.empty_cache()

    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW([
        {'params': model.vivit.parameters(),      'lr': VIDEO_LR_BACKBONE},
        {'params': model.embed_proj.parameters(), 'lr': VIDEO_LR_HEAD},
        {'params': model.classifier.parameters(), 'lr': VIDEO_LR_HEAD},
    ], weight_decay=VIDEO_WEIGHT_DECAY)

    steps_fine = len(train_dl_fine)
    scheduler  = WarmupCosineScheduler(
        optimizer,
        warmup_steps=steps_fine * 2,
        total_steps=steps_fine * VIDEO_EPOCHS_FINE,
    )

    patience   = 8
    no_improve = 0

    for epoch in range(1, VIDEO_EPOCHS_FINE + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch_finetune(
            model, train_dl_fine, optimizer, scheduler, criterion,
            device, scaler, mixup_alpha=0.3
        )
        vl_loss, vl_acc, preds, labels = evaluate(model, val_dl, val_criterion, device)
        dt = time.time() - t0

        ep_global = VIDEO_EPOCHS_WARM + epoch
        print(f"Ep {ep_global:02d}/{VIDEO_EPOCHS_WARM + VIDEO_EPOCHS_FINE} [fine] | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"LR: {scheduler.get_lr()[0]:.2e} | Time: {dt:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save({
                'epoch': ep_global,
                'phase': 'finetune',
                'model_state_dict': model.state_dict(),
                'val_acc': vl_acc,
                'class_names': EMOTION_CLASSES,
            }, best_path)
            print(f"  Best saved (val_acc={vl_acc:.4f})")
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == VIDEO_EPOCHS_FINE:
            metrics = evaluate_predictions(labels, preds)
            print(metrics['report'])

        if no_improve >= patience:
            print(f"\nEarly stopping la epoca {ep_global} (patience={patience})")
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

    _, test_acc, test_preds, test_labels = evaluate(model, test_dl, val_criterion, device)
    metrics = evaluate_predictions(test_labels, test_preds)
    print_metrics(metrics, "VIDEO MODEL — Test Set Results")

    np.save(RESULTS_DIR / "video_test_preds.npy",  np.array(test_preds))
    np.save(RESULTS_DIR / "video_test_labels.npy", np.array(test_labels))

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model salvat: {best_path}")
    print(f"Training complete!")


if __name__ == '__main__':
    main()
