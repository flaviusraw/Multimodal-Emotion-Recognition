"""Fine-tuning Wav2Vec2 / HuBERT pe CREMA-D pentru Speech Emotion Recognition (SER).

Porneste dintr-un checkpoint SELF-SUPERVISED (fara labels de emotie), ex:

  facebook/wav2vec2-base                           (~95M, rapid)
  facebook/wav2vec2-large                          (~317M)
  jonatasgrosman/wav2vec2-large-xlsr-53-english    (~317M, multilingual SSL)
  facebook/hubert-base-ls960                       (~95M, HuBERT pe engleza)
  TencentGameMate/chinese-hubert-base              (~95M, HuBERT pe chineza)

Invata sa clasifice pe 6 emotii CREMA-D: angry, disgust, fear, happy, neutral, sad.

CARACTERISTICI:
  Split speaker-independent implicit 75/15/10 (pe cei 91 de actori CREMA-D)
  Learning rate diferentiat pe encoder vs head
  Mixed precision BFloat16 (optim pe H100)
  Augmentari pe waveform (zgomot gaussian, time shift, gain)
  Early stopping pe val F1-macro
  Salveaza checkpoint HuggingFace-compatible (load cu from_pretrained)
  Auto-detect arhitectura (Wav2Vec2 sau HuBERT) din numele checkpoint-ului
  [NOU] Afisare metrici test la fiecare epoca

EXEMPLU DE RULARE (Wav2Vec2):
  python train_wav2vec2_cremad.py \
    --crema-d /export/home/.../crema/AudioWAV \
    --base-model facebook/wav2vec2-base \
    --out-dir ./runs/wav2vec2_base_cremad \
    --epochs 15 --batch-size 16 --augment

EXEMPLU DE RULARE (HuBERT chinezesc - cross-lingual transfer):
  python train_wav2vec2_cremad.py \
    --crema-d /export/home/.../crema/AudioWAV \
    --base-model TencentGameMate/chinese-hubert-base \
    --out-dir ./runs/chinese_hubert_cremad \
    --epochs 15 --batch-size 16 --augment

Autor: script pentru lucrarea de licenta (MELECON 2026) - Flavius Rau.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import torchaudio
    _HAVE_TORCHAUDIO = True
except ImportError:
    _HAVE_TORCHAUDIO = False

import librosa

from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    HubertModel,
    HubertPreTrainedModel,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ==============================================================================
# 1. Constante
# ==============================================================================

TARGET_SR = 16000
CANONICAL_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
CREMA_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# ==============================================================================
# 2. Arhitectura - compatibila HF cu classifier.dense + classifier.out_proj
# ==============================================================================

class Wav2Vec2ClassificationHead(nn.Module):
    """Head: dense -> tanh -> dropout -> out_proj. Standard in SER-ul cu wav2vec2."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(config, "final_dropout", 0.1))
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 body + mean-pool peste timp + ClassificationHead."""
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = getattr(config, "pooling_mode", "mean")
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.post_init()

    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def _pool(hidden: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "mean":
            return hidden.mean(dim=1)
        if mode == "max":
            return hidden.max(dim=1).values
        raise ValueError(f"Pooling necunoscut: {mode}")

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        out = self.wav2vec2(input_values, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, self.pooling_mode)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class HubertForSpeechClassification(HubertPreTrainedModel):
    """HuBERT body + mean-pool peste timp + ClassificationHead."""
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = getattr(config, "pooling_mode", "mean")
        self.hubert = HubertModel(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.post_init()

    def freeze_feature_encoder(self):
        self.hubert.feature_extractor._freeze_parameters()

    @staticmethod
    def _pool(hidden: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "mean":
            return hidden.mean(dim=1)
        if mode == "max":
            return hidden.max(dim=1).values
        raise ValueError(f"Pooling necunoscut: {mode}")

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        out = self.hubert(input_values, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, self.pooling_mode)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def detect_model_type(base_model: str, config) -> str:
    mt = getattr(config, "model_type", "").lower()
    if mt in ("wav2vec2", "hubert"):
        return mt
    low = base_model.lower()
    if "hubert" in low:
        return "hubert"
    if "wav2vec2" in low or "wav2vec" in low:
        return "wav2vec2"
    return "wav2vec2"


def build_model_for_ser(base_model: str, config):
    model_type = detect_model_type(base_model, config)
    if model_type == "hubert":
        model = HubertForSpeechClassification.from_pretrained(
            base_model, config=config, ignore_mismatched_sizes=True,
        )
        print(f"[MODEL] Arhitectura detectata: HuBERT -> HubertForSpeechClassification")
    else:
        model = Wav2Vec2ForSpeechClassification.from_pretrained(
            base_model, config=config, ignore_mismatched_sizes=True,
        )
        print(f"[MODEL] Arhitectura detectata: Wav2Vec2 -> Wav2Vec2ForSpeechClassification")
    return model, model_type


def load_best_model(out_dir: Path, base_model: str):
    config = AutoConfig.from_pretrained(out_dir / "best_model")
    model_type = detect_model_type(base_model, config)
    if model_type == "hubert":
        return HubertForSpeechClassification.from_pretrained(out_dir / "best_model")
    return Wav2Vec2ForSpeechClassification.from_pretrained(out_dir / "best_model")

# ==============================================================================
# 3. Data loading + split speaker-independent
# ==============================================================================

def parse_crema_d(root: Path) -> List[Tuple[Path, str, str]]:
    samples: List[Tuple[Path, str, str]] = []
    for wav in sorted(root.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 4:
            continue
        speaker_id = parts[0]
        code = parts[2].upper()
        if code in CREMA_MAP:
            samples.append((wav, CREMA_MAP[code], speaker_id))
    return samples


def speaker_independent_split(
    samples: List[Tuple[Path, str, str]],
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[Dict[str, List], Dict[str, List[str]]]:
    speakers = sorted(set(s[2] for s in samples))
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n_test = max(1, int(round(len(speakers) * test_frac)))
    n_val = max(1, int(round(len(speakers) * val_frac)))
    test_sp = set(speakers[:n_test])
    val_sp = set(speakers[n_test:n_test + n_val])
    train_sp = set(speakers[n_test + n_val:])

    splits: Dict[str, List] = {"train": [], "val": [], "test": []}
    for s in samples:
        if s[2] in train_sp:
            splits["train"].append(s)
        elif s[2] in val_sp:
            splits["val"].append(s)
        else:
            splits["test"].append(s)

    split_speakers = {
        "train": sorted(train_sp),
        "val":   sorted(val_sp),
        "test":  sorted(test_sp),
    }
    return splits, split_speakers


def same_speaker_split(
    samples: List[Tuple[Path, str, str]],
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[Dict[str, List], Dict[str, List[str]]]:
    buckets: Dict[Tuple[str, str], List[Tuple[Path, str, str]]] = {}
    for s in samples:
        _, emo, sp = s
        buckets.setdefault((sp, emo), []).append(s)

    rng = random.Random(seed)
    splits: Dict[str, List] = {"train": [], "val": [], "test": []}

    for (sp, emo), items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        if n_val + n_test >= n:
            if n_test > 0:
                n_test = max(0, n - n_val - 1)
            if n_val > 0:
                n_val = max(0, n - n_test - 1)
        splits["test"].extend(items[:n_test])
        splits["val"].extend(items[n_test:n_test + n_val])
        splits["train"].extend(items[n_test + n_val:])

    split_speakers = {
        "train": sorted(set(s[2] for s in splits["train"])),
        "val":   sorted(set(s[2] for s in splits["val"])),
        "test":  sorted(set(s[2] for s in splits["test"])),
    }
    return splits, split_speakers


def load_waveform(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    if _HAVE_TORCHAUDIO:
        try:
            wav, sr = torchaudio.load(str(path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            return wav.squeeze(0).numpy().astype(np.float32)
        except Exception:
            pass
    wav, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return wav.astype(np.float32)


class CremaDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, str, str]],
        label2id: Dict[str, int],
        max_seconds: float = 6.0,
        augment: bool = False,
        target_sr: int = TARGET_SR,
    ):
        self.samples = samples
        self.label2id = label2id
        self.max_len = int(max_seconds * target_sr)
        self.augment = augment
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, _ = self.samples[idx]
        wav = load_waveform(path, self.target_sr)

        if self.augment:
            if random.random() < 0.5:
                noise = np.random.randn(len(wav)).astype(np.float32) * 0.005
                wav = wav + noise
            if random.random() < 0.3:
                shift = random.randint(-1600, 1600)
                wav = np.roll(wav, shift)
            if random.random() < 0.3:
                gain = 10 ** (random.uniform(-0.15, 0.15))
                wav = wav * gain

        if wav.shape[0] > self.max_len:
            if self.augment:
                start = random.randint(0, wav.shape[0] - self.max_len)
            else:
                start = (wav.shape[0] - self.max_len) // 2
            wav = wav[start:start + self.max_len]

        return wav, self.label2id[label], str(path)


def make_collate(feature_extractor: Wav2Vec2FeatureExtractor):
    def collate(batch):
        waves = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        paths = [b[2] for b in batch]
        inputs = feature_extractor(
            waves,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        return {
            "input_values": inputs["input_values"],
            "attention_mask": inputs.get("attention_mask"),
            "labels": labels,
            "paths": paths,
        }
    return collate

# ==============================================================================
# 4. Evaluare
# ==============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    use_bf16: bool = True,
) -> Dict:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    n = 0

    for batch in loader:
        input_values = batch["input_values"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
        else:
            out = model(input_values=input_values, attention_mask=attention_mask, labels=labels)

        total_loss += out.loss.item() * labels.size(0)
        n += labels.size(0)
        preds = out.logits.argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    label_names = [id2label[i] for i in range(len(id2label))]
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "f1_weighted": float(f1w),
        "preds": all_preds,
        "labels": all_labels,
        "label_names": label_names,
    }

# ==============================================================================
# 5. Main
# ==============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tuning Wav2Vec2 pe CREMA-D (SER).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Date
    p.add_argument("--crema-d", required=True, type=str,
                   help="Calea catre CREMA-D/AudioWAV")
    p.add_argument("--out-dir", default="./runs/wav2vec2_cremad", type=str)

    # Model de pornire
    p.add_argument("--base-model", default="facebook/wav2vec2-base", type=str,
                   help="Checkpoint de pornire (SSL, fara emotii).")
    p.add_argument("--pooling-mode", default="mean", choices=["mean", "max"])
    p.add_argument("--final-dropout", default=0.1, type=float)
    p.add_argument("--freeze-feature-encoder", action="store_true",
                   help="Ingheata feature extractor-ul convolutional (recomandat).")

    # Training
    p.add_argument("--epochs", default=15, type=int)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--grad-accum", default=1, type=int)
    p.add_argument("--lr-encoder", default=1e-5, type=float)
    p.add_argument("--lr-head", default=1e-3, type=float)
    p.add_argument("--weight-decay", default=0.01, type=float)
    p.add_argument("--warmup-ratio", default=0.1, type=float)
    p.add_argument("--max-grad-norm", default=1.0, type=float)
    p.add_argument("--max-seconds", default=6.0, type=float)
    p.add_argument("--patience", default=5, type=int,
                   help="Early stopping: nr de epoci fara imbunatatire la val F1.")
    p.add_argument("--augment", action="store_true",
                   help="Activeaza augmentarile de waveform la train.")

    # Split
    p.add_argument("--split-mode", default="same-speaker",
                   choices=["same-speaker", "speaker-independent"],
                   help=(
                       "same-speaker (default): toti cei 91 de actori apar in train/val/test, "
                       "fisierele lor se impart 75/15/10 stratificat pe emotie. "
                       "speaker-independent: actorii insisi se impart (test real de generalizare)."
                   ))
    p.add_argument("--val-frac", default=0.15, type=float)
    p.add_argument("--test-frac", default=0.10, type=float)
    p.add_argument("--seed", default=42, type=int)

    # Diverse
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--no-bf16", action="store_true",
                   help="Dezactiveaza mixed precision BFloat16.")
    p.add_argument("--log-every", default=50, type=int,
                   help="La cate step-uri sa printeze loss-ul curent.")
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (device.type == "cuda") and (not args.no_bf16)
    print(f"[ENV] Device: {device}")
    if device.type == "cuda":
        print(f"[ENV] GPU:    {torch.cuda.get_device_name(0)}")
    print(f"[ENV] BFloat16 mixed precision: {use_bf16}")

    # --- Data ---
    samples = parse_crema_d(Path(args.crema_d))
    print(f"\n[DATA] Total fisiere CREMA-D: {len(samples)}")
    all_speakers = sorted(set(s[2] for s in samples))
    print(f"[DATA] Total actori: {len(all_speakers)}")

    if args.split_mode == "speaker-independent":
        print(f"[DATA] Split mode: SPEAKER-INDEPENDENT "
              f"(actori diferiti in train/val/test)")
        splits, split_speakers = speaker_independent_split(
            samples, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
    else:
        print(f"[DATA] Split mode: SAME-SPEAKER "
              f"(toti {len(all_speakers)} actori in toate splitur-ile, "
              f"fisierele impartite {int((1 - args.val_frac - args.test_frac)*100)}/"
              f"{int(args.val_frac*100)}/{int(args.test_frac*100)} stratificat pe emotie)")
        splits, split_speakers = same_speaker_split(
            samples, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )

    for k, v in splits.items():
        print(f"[DATA]   {k}: {len(v)} esantioane ({len(split_speakers[k])} actori)")
        distro = Counter(s[1] for s in v)
        print(f"[DATA]     distributie: {dict(distro)}")

    with open(out_dir / "split_speakers.json", "w") as f:
        json.dump(split_speakers, f, indent=2)

    # --- Labels ---
    label2id = {lab: i for i, lab in enumerate(CANONICAL_EMOTIONS)}
    id2label = {i: lab for lab, i in label2id.items()}

    # --- Feature extractor ---
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model)
    collate_fn = make_collate(feature_extractor)

    # --- Datasets + loaders ---
    train_ds = CremaDataset(splits["train"], label2id,
                            max_seconds=args.max_seconds, augment=args.augment)
    val_ds   = CremaDataset(splits["val"],   label2id,
                            max_seconds=args.max_seconds, augment=False)
    test_ds  = CremaDataset(splits["test"],  label2id,
                            max_seconds=args.max_seconds, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # --- Model ---
    print(f"\n[MODEL] Pornesc de la: {args.base_model}")
    config = AutoConfig.from_pretrained(args.base_model)
    config.num_labels = len(CANONICAL_EMOTIONS)
    config.label2id = label2id
    config.id2label = {str(k): v for k, v in id2label.items()}
    config.pooling_mode = args.pooling_mode
    config.final_dropout = args.final_dropout

    model, model_type = build_model_for_ser(args.base_model, config)
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()
        print("[MODEL] Feature encoder CNN INGHETAT.")

    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Parametri totali: {n_params/1e6:.1f}M | antrenabili: {n_trainable/1e6:.1f}M")

    # --- Optimizer: LR diferentiat pentru encoder vs head ---
    encoder_params = [p for n, p in model.named_parameters()
                      if (not n.startswith("classifier.")) and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if n.startswith("classifier.") and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr_encoder, "weight_decay": args.weight_decay},
            {"params": head_params,    "lr": args.lr_head,    "weight_decay": args.weight_decay},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"[OPT]   Steps totali: {total_steps}, warmup: {warmup_steps}")
    print(f"[OPT]   LR encoder: {args.lr_encoder} | LR head: {args.lr_head}")

    # --- Loop de antrenare ---
    log_path = out_dir / "training.log"
    log_file = open(log_path, "w")

    def log(msg: str):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\nConfig: {json.dumps(vars(args), indent=2)}")

    best_f1 = -1.0
    best_epoch = 0
    patience_left = args.patience
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0
        train_correct = 0  # NOU: pentru acuratete la train
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"]
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                    loss = out.loss / args.grad_accum
            else:
                out = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                loss = out.loss / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += out.loss.item() * labels.size(0)
            seen += labels.size(0)
            # NOU: numara predictiile corecte la train
            train_correct += (out.logits.detach().argmax(dim=-1) == labels).sum().item()

            if (step + 1) % args.log_every == 0:
                lr_enc = optimizer.param_groups[0]["lr"]
                lr_head = optimizer.param_groups[1]["lr"]
                log(f"  [ep {epoch} step {step+1:4d}/{len(train_loader)}]  "
                    f"loss={running_loss/seen:.4f}  acc={train_correct/seen:.4f}  "
                    f"lr_enc={lr_enc:.2e}  lr_head={lr_head:.2e}")

        train_loss = running_loss / max(seen, 1)
        train_acc = train_correct / max(seen, 1)  # NOU

        # --- Validation ---
        val = evaluate(model, val_loader, device, id2label, use_bf16=use_bf16)

        # ── NOU: Test la fiecare epoca ──────────────────────────────────────
        test_ep = evaluate(model, test_loader, device, id2label, use_bf16=use_bf16)
        # ───────────────────────────────────────────────────────────────────

        elapsed = time.time() - t0
        log(
            f"[ep {epoch:3d}]  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
            f"val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.4f}  "
            f"val_f1m={val['f1_macro']:.4f}  val_f1w={val['f1_weighted']:.4f}  |  "
            f"test_loss={test_ep['loss']:.4f}  test_acc={test_ep['accuracy']:.4f}  "
            f"test_f1m={test_ep['f1_macro']:.4f}  test_f1w={test_ep['f1_weighted']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,  # NOU
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_f1_macro": val["f1_macro"],
            "val_f1_weighted": val["f1_weighted"],
            # ── NOU: test metrics in history ────────────────────────────────
            "test_loss": test_ep["loss"],
            "test_accuracy": test_ep["accuracy"],
            "test_f1_macro": test_ep["f1_macro"],
            "test_f1_weighted": test_ep["f1_weighted"],
            # ────────────────────────────────────────────────────────────────
            "elapsed_s": elapsed,
        })

        # --- Early stopping pe f1 macro ---
        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            best_epoch = epoch
            patience_left = args.patience
            save_dir = out_dir / "best_model"
            save_dir.mkdir(exist_ok=True)
            model.save_pretrained(save_dir)
            feature_extractor.save_pretrained(save_dir)
            log(f"           -> NEW BEST val_f1_macro={best_f1:.4f}. Salvat in {save_dir}")
        else:
            patience_left -= 1
            log(f"           -> fara imbunatatire (patience ramas: {patience_left})")
            if patience_left <= 0:
                log(f"           -> Early stopping la epoca {epoch}.")
                break

    # Salvam history-ul (include acum si metricile de test per epoca)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # --- Evaluare finala pe test cu cel mai bun checkpoint ---
    log(f"\n[TEST] Incarc cel mai bun checkpoint (epoca {best_epoch}, val_f1m={best_f1:.4f})...")
    best_model = load_best_model(out_dir, args.base_model).to(device)

    test = evaluate(best_model, test_loader, device, id2label, use_bf16=use_bf16)

    log(
        f"\n[TEST FINAL (best checkpoint)]  accuracy={test['accuracy']:.4f}  "
        f"f1_macro={test['f1_macro']:.4f}  f1_weighted={test['f1_weighted']:.4f}"
    )

    report = classification_report(
        test["labels"], test["preds"],
        target_names=test["label_names"],
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(
        test["labels"], test["preds"],
        labels=list(range(len(CANONICAL_EMOTIONS))),
    )

    log("\n[TEST] Classification report:\n" + report)
    log(f"[TEST] Confusion matrix (ordine: {test['label_names']}):\n{cm}")

    with open(out_dir / "test_results.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_f1_macro": best_f1,
            "test_accuracy": test["accuracy"],
            "test_f1_macro": test["f1_macro"],
            "test_f1_weighted": test["f1_weighted"],
            "label_order": test["label_names"],
            "confusion_matrix": cm.tolist(),
            "test_speakers": split_speakers["test"],
        }, f, indent=2, ensure_ascii=False)

    np.save(out_dir / "test_confusion_matrix.npy", cm)

    log(f"\nGata. Cel mai bun checkpoint: {(out_dir / 'best_model').resolve()}")
    log(f"Log complet:                   {log_path.resolve()}")
    log_file.close()


if __name__ == "__main__":
    main()
