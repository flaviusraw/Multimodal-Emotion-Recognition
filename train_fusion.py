"""
train_fusion.py — Weighted Late Fusion (Fig. 3 din articol)

Etape:
  1. Incarca modelele video si audio pre-antrenate
  2. Extrage embedding-uri din CREMA-D (perechi video+audio)
  3. Antreneaza modulul de fuziune cu ponderi learnable
  4. Evalueaza pe test set
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import *
from models import VideoViT, AudioTransformer, WeightedLateFusion
from datasets import (
    get_video_transforms, extract_mfcc_features,
    EMOTION_TO_IDX,
)
from train_utils import (
    FocalLoss, WarmupCosineScheduler,
    evaluate_predictions, print_metrics,
)
from PIL import Image


def load_pretrained_video_model(device):
    """Incarca modelul video ViT pre-antrenat."""
    path = MODELS_DIR / "video_vit_best.pt"
    if not path.exists():
        print(f"EROARE: {path} nu exista. Ruleaza mai intai: python train_video.py")
        sys.exit(1)

    model = VideoViT(num_classes=NUM_CLASSES, embed_dim=VIDEO_EMBED_DIM,
                     dropout=VIDEO_DROPOUT, pretrained=False).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Video model incarcat (val_acc={ckpt['val_acc']:.4f}, epoca {ckpt['epoch']})")
    return model


def load_pretrained_audio_model(device):
    """Incarca modelul audio Transformer pre-antrenat."""
    path = MODELS_DIR / "audio_transformer_best.pt"
    if not path.exists():
        print(f"EROARE: {path} nu exista. Ruleaza mai intai: python train_audio.py")
        sys.exit(1)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt['audio_config']

    model = AudioTransformer(
        input_dim=cfg['input_dim'],
        num_classes=NUM_CLASSES,
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_layers=cfg['num_layers'],
        dim_ff=cfg['dim_ff'],
        dropout=AUDIO_DROPOUT,
        embed_dim=AUDIO_EMBED_DIM,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Audio model incarcat (val_acc={ckpt['val_acc']:.4f}, epoca {ckpt['epoch']})")
    return model, cfg['max_len']


@torch.no_grad()
def extract_embeddings(video_model, audio_model, df, device, audio_max_len):
    """
    Extrage embedding-uri video si audio pentru fiecare pereche din df.
    df trebuie sa contina coloanele: video_path, audio_path, emotion
    """
    video_tf = get_video_transforms(is_train=False)
    feature_dim = AUDIO_N_MFCC * 3

    video_embeddings = []
    audio_embeddings = []
    labels = []
    skipped = 0

    bar = tqdm(df.iterrows(), total=len(df), desc="  Extracting embeddings")
    for _, row in bar:
        # ── Video embedding ──
        try:
            img = Image.open(row['video_path']).convert('RGB')
            img_tensor = video_tf(img).unsqueeze(0).to(device)
            v_emb = video_model.get_embedding(img_tensor).cpu().numpy().squeeze()
        except Exception:
            skipped += 1
            continue

        # ── Audio embedding ──
        try:
            features = extract_mfcc_features(row['audio_path'])
            if features is None:
                skipped += 1
                continue

            # Pad/truncate
            T, F = features.shape
            if T < audio_max_len:
                features = np.vstack([features,
                    np.zeros((audio_max_len - T, F), dtype=np.float32)])
            else:
                features = features[:audio_max_len]

            # Normalizare
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True) + 1e-8
            features = (features - mean) / std

            audio_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            a_emb = audio_model.get_embedding(audio_tensor).cpu().numpy().squeeze()
        except Exception:
            skipped += 1
            continue

        video_embeddings.append(v_emb)
        audio_embeddings.append(a_emb)
        labels.append(EMOTION_TO_IDX[row['emotion']])

    if skipped > 0:
        print(f"  Skipped {skipped} samples")

    return (
        np.array(video_embeddings, dtype=np.float32),
        np.array(audio_embeddings, dtype=np.float32),
        np.array(labels, dtype=np.int64),
    )


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, video_emb, audio_emb, labels):
        self.video_emb = torch.tensor(video_emb, dtype=torch.float32)
        self.audio_emb = torch.tensor(audio_emb, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.video_emb[idx], self.audio_emb[idx], self.labels[idx]


def train_fusion_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for v_emb, a_emb, labels in loader:
        v_emb = v_emb.to(device, non_blocking=True)
        a_emb = a_emb.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(v_emb, a_emb)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_fusion(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for v_emb, a_emb, labels in loader:
        v_emb = v_emb.to(device, non_blocking=True)
        a_emb = a_emb.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(v_emb, a_emb)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[FUSION] Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Incarca modele pre-antrenate ─────────────────────────────────────
    print("\nIncarcarea modelelor pre-antrenate...")
    video_model = load_pretrained_video_model(device)
    audio_model, audio_max_len = load_pretrained_audio_model(device)

    # ── Incarca perechi fusion ───────────────────────────────────────────
    train_csv = PROJECT_DIR / "fusion_train.csv"
    val_csv   = PROJECT_DIR / "fusion_val.csv"
    test_csv  = PROJECT_DIR / "fusion_test.csv"

    for f in [train_csv, val_csv, test_csv]:
        if not f.exists():
            print(f"EROARE: {f} nu exista. Ruleaza mai intai: python prepare_data.py")
            sys.exit(1)

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    print(f"\n  Fusion pairs: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # ── Extrage embedding-uri ────────────────────────────────────────────
    emb_cache = MODELS_DIR / "fusion_embeddings.npz"

    if emb_cache.exists():
        print("\nIncarcarea embedding-urilor din cache...")
        data = np.load(emb_cache)
        tr_v, tr_a, tr_y = data['tr_v'], data['tr_a'], data['tr_y']
        vl_v, vl_a, vl_y = data['vl_v'], data['vl_a'], data['vl_y']
        te_v, te_a, te_y = data['te_v'], data['te_a'], data['te_y']
    else:
        print("\nExtragere embedding-uri (se ruleaza o singura data)...")
        print("  Train set:")
        tr_v, tr_a, tr_y = extract_embeddings(
            video_model, audio_model, train_df, device, audio_max_len)
        print("  Val set:")
        vl_v, vl_a, vl_y = extract_embeddings(
            video_model, audio_model, val_df, device, audio_max_len)
        print("  Test set:")
        te_v, te_a, te_y = extract_embeddings(
            video_model, audio_model, test_df, device, audio_max_len)

        np.savez(emb_cache,
                 tr_v=tr_v, tr_a=tr_a, tr_y=tr_y,
                 vl_v=vl_v, vl_a=vl_a, vl_y=vl_y,
                 te_v=te_v, te_a=te_a, te_y=te_y)
        print(f"  Cache salvat: {emb_cache}")

    print(f"\n  Embedding shapes: video={tr_v.shape[1]}, audio={tr_a.shape[1]}")
    print(f"  Train: {len(tr_y)} | Val: {len(vl_y)} | Test: {len(te_y)}")

    # ── Dataloaders fuziune ──────────────────────────────────────────────
    train_ds = FusionDataset(tr_v, tr_a, tr_y)
    val_ds   = FusionDataset(vl_v, vl_a, vl_y)
    test_ds  = FusionDataset(te_v, te_a, te_y)

    train_dl = DataLoader(train_ds, batch_size=FUSION_BATCH_SIZE,
                          shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=FUSION_BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=FUSION_BATCH_SIZE, shuffle=False)

    # ── Model fuziune ────────────────────────────────────────────────────
    fusion_model = WeightedLateFusion(
        video_embed_dim=VIDEO_EMBED_DIM,
        audio_embed_dim=AUDIO_EMBED_DIM,
        hidden_dim=FUSION_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=FUSION_DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in fusion_model.parameters())
    print(f"\n  Fusion model params: {total_params:,}")

    # Class weights
    cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=tr_y)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1, weight=class_weights)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        fusion_model.parameters(),
        lr=FUSION_LR,
        weight_decay=FUSION_WEIGHT_DECAY,
    )

    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * FUSION_EPOCHS
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=steps_per_epoch * 3,
        total_steps=total_steps,
    )

    # ── Antrenare fuziune ────────────────────────────────────────────────
    best_val_acc = 0.0
    best_path = MODELS_DIR / "fusion_best.pt"
    patience = 10
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"ANTRENARE WEIGHTED LATE FUSION ({FUSION_EPOCHS} epoci)")
    print(f"{'='*60}")

    for epoch in range(1, FUSION_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_fusion_epoch(
            fusion_model, train_dl, optimizer, criterion, device)

        # Step scheduler
        for _ in range(steps_per_epoch):
            scheduler.step()

        vl_loss, vl_acc, preds, labels = eval_fusion(
            fusion_model, val_dl, val_criterion, device)
        dt = time.time() - t0

        # Afiseaza ponderile curente
        weights = fusion_model.get_weights()

        print(f"Ep {epoch:02d}/{FUSION_EPOCHS} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"W_video={weights['video_weight']:.3f} "
              f"W_audio={weights['audio_weight']:.3f} | "
              f"Time: {dt:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': fusion_model.state_dict(),
                'val_acc': vl_acc,
                'weights': weights,
                'class_names': EMOTION_CLASSES,
            }, best_path)
            print(f"  ✓ Best saved (val_acc={vl_acc:.4f})")
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == FUSION_EPOCHS:
            metrics = evaluate_predictions(labels, preds)
            print(metrics['report'])

        if no_improve >= patience:
            print(f"\nEarly stopping la epoca {epoch} (patience={patience})")
            break

    # ══════════════════════════════════════════════════════════════════════
    # EVALUARE FINALA PE TEST SET
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("EVALUARE FINALA — TOATE MODELELE PE TEST SET")
    print(f"{'='*60}")

    # 1) Doar Video (pe perechi)
    print("\n--- Video Only (pe perechile de test) ---")
    video_only_preds = []
    video_model.eval()
    video_tf = get_video_transforms(is_train=False)
    for _, row in test_df.iterrows():
        try:
            img = Image.open(row['video_path']).convert('RGB')
            img_t = video_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = video_model(img_t).argmax(1).item()
            video_only_preds.append(pred)
        except Exception:
            video_only_preds.append(0)  # fallback

    v_metrics = evaluate_predictions(te_y[:len(video_only_preds)], video_only_preds)
    print_metrics(v_metrics, "VIDEO ONLY — Test")

    # 2) Doar Audio (pe perechi)
    print("\n--- Audio Only (pe perechile de test) ---")
    audio_only_preds = []
    audio_model.eval()
    feature_dim = AUDIO_N_MFCC * 3
    for _, row in test_df.iterrows():
        try:
            features = extract_mfcc_features(row['audio_path'])
            if features is None:
                audio_only_preds.append(0)
                continue
            T, F = features.shape
            if T < audio_max_len:
                features = np.vstack([features,
                    np.zeros((audio_max_len - T, F), dtype=np.float32)])
            else:
                features = features[:audio_max_len]
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True) + 1e-8
            features = (features - mean) / std
            t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = audio_model(t).argmax(1).item()
            audio_only_preds.append(pred)
        except Exception:
            audio_only_preds.append(0)

    a_metrics = evaluate_predictions(te_y[:len(audio_only_preds)], audio_only_preds)
    print_metrics(a_metrics, "AUDIO ONLY — Test")

    # 3) Fuziune
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    fusion_model.load_state_dict(ckpt['model_state_dict'])
    print(f"\nFusion model incarcat (val_acc={ckpt['val_acc']:.4f})")

    _, test_acc, test_preds, test_labels = eval_fusion(
        fusion_model, test_dl, val_criterion, device)
    f_metrics = evaluate_predictions(test_labels, test_preds)
    print_metrics(f_metrics, "WEIGHTED LATE FUSION — Test")

    # Ponderi finale
    final_weights = fusion_model.get_weights()
    print(f"\nPonderi finale invatate:")
    print(f"  Video: {final_weights['video_weight']:.4f}")
    print(f"  Audio: {final_weights['audio_weight']:.4f}")

    # Salvare rezultate
    np.save(RESULTS_DIR / "fusion_test_preds.npy", np.array(test_preds))
    np.save(RESULTS_DIR / "fusion_test_labels.npy", np.array(test_labels))

    # ── Rezumat comparativ ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("REZUMAT COMPARATIV")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"{'-'*57}")
    print(f"{'Video (ViT)':<25} {v_metrics['accuracy']*100:>7.2f}% "
          f"{v_metrics['macro_precision']*100:>7.2f}% "
          f"{v_metrics['macro_recall']*100:>7.2f}% "
          f"{v_metrics['macro_f1']*100:>7.2f}%")
    print(f"{'Audio (Transformer)':<25} {a_metrics['accuracy']*100:>7.2f}% "
          f"{a_metrics['macro_precision']*100:>7.2f}% "
          f"{a_metrics['macro_recall']*100:>7.2f}% "
          f"{a_metrics['macro_f1']*100:>7.2f}%")
    print(f"{'Fusion (Weighted)':<25} {f_metrics['accuracy']*100:>7.2f}% "
          f"{f_metrics['macro_precision']*100:>7.2f}% "
          f"{f_metrics['macro_recall']*100:>7.2f}% "
          f"{f_metrics['macro_f1']*100:>7.2f}%")
    print(f"{'='*60}")

    print(f"\nBest fusion val accuracy: {best_val_acc:.4f}")
    print(f"Model salvat: {best_path}")
    print("DONE!")


if __name__ == '__main__':
    main()
