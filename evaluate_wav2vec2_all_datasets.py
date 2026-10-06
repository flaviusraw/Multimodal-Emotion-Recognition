"""
Evaluarea modelului pre-antrenat Wav2Vec2 (Yassmen/Wav2Vec2_Fine_tuned_on_CremaD_Speech_Emotion_Recognition)
pe toate seturile audio de pe server: CREMA-D, RAVDESS, SAVEE, TESS.

Modelul are 6 emoții: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'].
Pentru seturile care au emoții suplimentare (surprise, calm), acele fișiere sunt
excluse automat din evaluare (modelul nu le poate prezice).

Folosire tipică:
    python evaluate_wav2vec2_all_datasets.py \
        --crema-d /cale/catre/CREMA-D/AudioWAV \
        --ravdess /cale/catre/RAVDESS \
        --savee   /cale/catre/SAVEE/AudioData \
        --tess    /cale/catre/TESS \
        --out-dir ./rezultate_wav2vec2_yassmen \
        --batch-size 16

Dacă lipsește o cale, setul respectiv este sărit automat.

Autor: script generat pentru lucrarea de licență (MELECON 2026) - Flavius Rau.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Audio loading: preferam torchaudio (rapid), cadem pe librosa daca nu merge
try:
    import torchaudio
    _HAVE_TORCHAUDIO = True
except Exception:
    _HAVE_TORCHAUDIO = False

import librosa

from transformers import (
    AutoConfig,
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# -------------------------------------------------------------------------
# 1. Configurare model + etichete
# -------------------------------------------------------------------------

MODEL_ID = "Yassmen/Wav2Vec2_Fine_tuned_on_CremaD_Speech_Emotion_Recognition"
TARGET_SR = 16000

# Ordinea canonica a emotiilor asa cum e documentata in README-ul modelului.
# Daca modelul are id2label in config, il folosim pe acela; altfel cadem pe
# aceasta ordine alfabetica (consistenta cu README).
CANONICAL_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# Normalizare etichete scurte/variante -> forma canonica folosita de parserele noastre.
# Modelul Yassmen foloseste coduri de 3 litere (ang, dis, fea, hap, neu, sad).
SHORT_TO_LONG: Dict[str, str] = {
    "ang": "angry", "anger": "angry", "angry": "angry",
    "dis": "disgust", "disgust": "disgust",
    "fea": "fear", "fearful": "fear", "fear": "fear",
    "hap": "happy", "happy": "happy", "happiness": "happy",
    "neu": "neutral", "neutral": "neutral",
    "sad": "sad", "sadness": "sad",
    "sur": "surprise", "surprised": "surprise", "surprise": "surprise",
    "cal": "calm", "calm": "calm",
}


# -------------------------------------------------------------------------
# 1.b Arhitectura custom (compatibila cu checkpoint-ul Yassmen / ehcalabres / harshit345)
# -------------------------------------------------------------------------

class Wav2Vec2ClassificationHead(nn.Module):
    """
    Head-ul de clasificare folosit de modelele SER pe baza Wav2Vec2 de pe HF
    (Yassmen, ehcalabres, harshit345). Structura: dense -> tanh -> out_proj.
    Diferit de Wav2Vec2ForSequenceClassification standard care are un simplu
    projector + linear!
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(config, "final_dropout", 0.1))
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """
    Arhitectura compatibila cu checkpoint-ul Yassmen:
        wav2vec2 body + mean-pool peste timp + ClassificationHead (dense+out_proj).
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = getattr(config, "pooling_mode", "mean")
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.post_init()

    @staticmethod
    def _pool(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor], mode: str):
        if attention_mask is None:
            if mode == "mean":
                return hidden_states.mean(dim=1)
            if mode == "sum":
                return hidden_states.sum(dim=1)
            if mode == "max":
                return hidden_states.max(dim=1).values
            raise ValueError(f"Pooling mode necunoscut: {mode}")
        # Pooling mascat (ignora padding-ul)
        mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        if mode == "mean":
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            return summed / counts
        if mode == "sum":
            return (hidden_states * mask).sum(dim=1)
        if mode == "max":
            hidden_states = hidden_states.masked_fill(mask == 0, float("-inf"))
            return hidden_states.max(dim=1).values
        raise ValueError(f"Pooling mode necunoscut: {mode}")

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]  # (B, T, H)

        # Atentie: lungimea de dupa convolutii e mai mica decat attention_mask original.
        # Ramane rezonabil sa aplicam mean-pool direct (asa e si in impl. originala
        # ehcalabres/harshit345 care nu downsample-ul mask-ul).
        pooled = self._pool(hidden_states, None, self.pooling_mode)

        logits = self.classifier(pooled)
        return SequenceClassifierOutput(logits=logits)

# Mapari CREMA-D (codurile din numele fisierelor -> eticheta canonica)
CREMA_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# Mapari RAVDESS (codul 01..08 din numele fisierelor)
# 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
RAVDESS_MAP = {
    "01": "neutral",
    "02": "calm",       # nu exista in model -> va fi filtrat
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",       # "fearful" -> "fear"
    "07": "disgust",
    "08": "surprise",   # nu exista in model -> va fi filtrat
}

# Mapari SAVEE - codul este prefixul literal al numelui fisierului
# a = anger, d = disgust, f = fear, h = happiness, n = neutral,
# sa = sadness, su = surprise
# Atentie: sa/su sunt 2 litere, deci trebuie verificate inainte de s single.
SAVEE_PREFIXES: List[Tuple[str, str]] = [
    ("sa", "sad"),
    ("su", "surprise"),  # va fi filtrat
    ("a",  "angry"),
    ("d",  "disgust"),
    ("f",  "fear"),
    ("h",  "happy"),
    ("n",  "neutral"),
]

# Mapari TESS - emotia apare ca ultim cuvant in numele fisierului (inainte de .wav)
# ex: OAF_back_angry.wav, YAF_ring_ps.wav (ps = pleasant surprise)
TESS_MAP = {
    "angry":   "angry",
    "disgust": "disgust",
    "fear":    "fear",
    "happy":   "happy",
    "neutral": "neutral",
    "sad":     "sad",
    "ps":      "surprise",   # pleasant surprise -> filtrat
}


# -------------------------------------------------------------------------
# 2. Parsere per dataset - returneaza lista (cale_fisier, emotie_adevarata)
# -------------------------------------------------------------------------

def parse_crema_d(root: Path) -> List[Tuple[Path, str]]:
    """
    CREMA-D: fisiere .wav cu nume SPEAKER_SENTENCE_EMOTION_INTENSITY.wav
    ex: 1001_DFA_ANG_XX.wav  -> emotie = ANG -> "angry"
    """
    samples: List[Tuple[Path, str]] = []
    for wav in sorted(root.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 4:
            continue
        code = parts[2].upper()
        if code in CREMA_MAP:
            samples.append((wav, CREMA_MAP[code]))
    return samples


def parse_ravdess(root: Path) -> List[Tuple[Path, str]]:
    """
    RAVDESS: cautam recursiv fisiere .wav cu nume de forma
    03-01-EE-II-SS-RR-AA.wav (doar modality=03 audio-only speech).
    Pozitia 3 (EE) = codul emotiei.
    """
    samples: List[Tuple[Path, str]] = []
    for wav in sorted(root.rglob("*.wav")):
        name = wav.stem
        parts = name.split("-")
        if len(parts) != 7:
            continue
        modality = parts[0]
        # 03 = audio-only. 01 = full-AV, 02 = video-only. Pastram doar audio-only.
        if modality != "03":
            continue
        emo_code = parts[2]
        if emo_code in RAVDESS_MAP:
            samples.append((wav, RAVDESS_MAP[emo_code]))
    return samples


def parse_savee(root: Path) -> List[Tuple[Path, str]]:
    """
    SAVEE: structura tipica AudioData/<speaker>/<prefix><index>.wav
    Prefixele: a, d, f, h, n, sa, su.
    """
    samples: List[Tuple[Path, str]] = []
    for wav in sorted(root.rglob("*.wav")):
        stem = wav.stem.lower()
        matched = None
        for prefix, emo in SAVEE_PREFIXES:
            if stem.startswith(prefix) and (
                len(stem) > len(prefix) and stem[len(prefix)].isdigit()
            ):
                matched = emo
                break
        if matched is not None:
            samples.append((wav, matched))
    return samples


def parse_tess(root: Path) -> List[Tuple[Path, str]]:
    """
    TESS: fisiere de forma <actor>_<cuvant>_<emotie>.wav, grupate in subfoldere.
    Emotie = ultimul token din nume.
    """
    samples: List[Tuple[Path, str]] = []
    for wav in sorted(root.rglob("*.wav")):
        stem = wav.stem.lower()
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        emo_token = parts[-1]
        if emo_token in TESS_MAP:
            samples.append((wav, TESS_MAP[emo_token]))
    return samples


PARSERS: Dict[str, Callable[[Path], List[Tuple[Path, str]]]] = {
    "CREMA-D": parse_crema_d,
    "RAVDESS": parse_ravdess,
    "SAVEE":   parse_savee,
    "TESS":    parse_tess,
}


# -------------------------------------------------------------------------
# 3. Dataset PyTorch pentru inferenta batched
# -------------------------------------------------------------------------

def load_waveform(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Incarca un fisier audio si il aduce la 16kHz mono (float32)."""
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
    # Fallback librosa
    wav, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return wav.astype(np.float32)


class AudioEmotionDataset(Dataset):
    """
    Returneaza (waveform_float32, eticheta_string, calea_fisierului) pentru
    fiecare esantion. Capeaza durata pentru a limita memoria GPU.
    """
    def __init__(
        self,
        samples: List[Tuple[Path, str]],
        max_seconds: float = 8.0,
        target_sr: int = TARGET_SR,
    ):
        self.samples = samples
        self.max_len = int(max_seconds * target_sr)
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        wav = load_waveform(path, self.target_sr)
        if wav.shape[0] > self.max_len:
            # Centram taierea (luam portiunea din mijloc)
            start = (wav.shape[0] - self.max_len) // 2
            wav = wav[start:start + self.max_len]
        return wav, label, str(path)


def collate_audio(batch):
    """Collate custom: intoarce liste (feature_extractor face padding-ul)."""
    waves = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    paths = [b[2] for b in batch]
    return waves, labels, paths


# -------------------------------------------------------------------------
# 4. Incarcarea modelului si inferenta
# -------------------------------------------------------------------------

@dataclass
class LoadedModel:
    model: torch.nn.Module
    feature_extractor: Wav2Vec2FeatureExtractor
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    mode: str   # "ctc" sau "seq_cls"


def load_yassmen_model(device: torch.device) -> LoadedModel:
    """
    Incarca modelul Yassmen folosind arhitectura custom Wav2Vec2ForSpeechClassification.

    IMPORTANT: checkpoint-ul are head-ul (classifier.dense + classifier.out_proj)
    care NU se potriveste cu Wav2Vec2ForSequenceClassification standard (acela are
    projector + classifier simplu). Daca incarcam cu AutoModelForAudioClassification
    primim un head initializat ALEATOR -> predictii aiurea.
    """
    print(f"[MODEL] Se descarca / incarca: {MODEL_ID}", flush=True)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    if not hasattr(config, "pooling_mode") or config.pooling_mode is None:
        config.pooling_mode = "mean"

    model: torch.nn.Module
    mode = "seq_cls"
    try:
        model = Wav2Vec2ForSpeechClassification.from_pretrained(MODEL_ID, config=config)
        print(
            "[MODEL] Incarcat cu arhitectura custom Wav2Vec2ForSpeechClassification "
            f"(pooling={config.pooling_mode}).",
            flush=True,
        )
    except Exception as e:
        print(
            f"[MODEL] Arhitectura custom a esuat ({e}). Incerc AutoModelForAudioClassification...",
            flush=True,
        )
        try:
            model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
            print("[MODEL] Incarcat ca AutoModelForAudioClassification (head-ul ar putea fi random!).", flush=True)
        except Exception:
            model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
            mode = "ctc"
            print("[MODEL] Fallback Wav2Vec2ForCTC (mediere peste timp).", flush=True)

    # Construim id2label / label2id si le normalizam
    id2label_raw: Dict[int, str] = {}
    if getattr(config, "id2label", None):
        id2label_raw = {int(k): str(v).lower().strip() for k, v in config.id2label.items()}

    # Normalizam forme scurte/lungi -> forma canonica
    id2label = {k: SHORT_TO_LONG.get(v, v) for k, v in id2label_raw.items()}

    # Fallback daca id2label e gol sau are chei generice
    if (not id2label) or any(str(v).startswith("label_") for v in id2label.values()):
        print(
            "[MODEL] id2label lipseste sau e generic; folosesc ordinea canonica "
            f"din README: {CANONICAL_EMOTIONS}",
            flush=True,
        )
        id2label = {i: lab for i, lab in enumerate(CANONICAL_EMOTIONS)}

    label2id = {v: k for k, v in id2label.items()}
    if id2label_raw and id2label_raw != id2label:
        print(f"[MODEL] id2label (raw din config) = {id2label_raw}", flush=True)
    print(f"[MODEL] id2label (normalizat)        = {id2label}", flush=True)

    model.eval()
    model.to(device)
    return LoadedModel(model, feature_extractor, id2label, label2id, mode)


@torch.no_grad()
def predict_batch(
    loaded: LoadedModel,
    waves: List[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """
    Ruleaza o inferenta batched. Intoarce matricea de probabilitati (B, num_labels).
    """
    inputs = loaded.feature_extractor(
        waves,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    kwargs = {"input_values": input_values}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    outputs = loaded.model(**kwargs)
    logits = outputs.logits  # (B, T, C) pentru CTC sau (B, C) pentru seq_cls

    if loaded.mode == "ctc" or logits.dim() == 3:
        # Mediem peste axa temporala
        logits = logits.mean(dim=1)

    probs = F.softmax(logits, dim=-1)
    return probs.cpu().numpy()


# -------------------------------------------------------------------------
# 5. Evaluare pe un dataset
# -------------------------------------------------------------------------

def evaluate_dataset(
    name: str,
    samples_all: List[Tuple[Path, str]],
    loaded: LoadedModel,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_seconds: float,
    out_dir: Path,
) -> Dict:
    """Ruleaza modelul pe un dataset si salveaza metricile."""
    model_labels = set(loaded.label2id.keys())

    # Filtram esantioanele a caror eticheta adevarata NU e in modelul nostru
    samples = [(p, lab) for (p, lab) in samples_all if lab in model_labels]
    skipped = len(samples_all) - len(samples)
    skipped_by_label = Counter(
        lab for (_, lab) in samples_all if lab not in model_labels
    )

    print(
        f"\n[{name}] Total fisiere gasite: {len(samples_all)} | "
        f"Evaluate: {len(samples)} | Excluse (emotii pe care modelul nu le cunoaste): {skipped}",
        flush=True,
    )
    if skipped_by_label:
        print(f"[{name}] Repartitie excluse: {dict(skipped_by_label)}", flush=True)
    if len(samples) == 0:
        print(f"[{name}] Nu raman esantioane evaluabile. Sar peste.", flush=True)
        return {"name": name, "n_evaluated": 0}

    # Distributie adevar pe cele 6 clase
    label_counts = Counter(lab for (_, lab) in samples)
    print(f"[{name}] Distributie etichete adevarate: {dict(label_counts)}", flush=True)

    dataset = AudioEmotionDataset(samples, max_seconds=max_seconds)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio,
        pin_memory=True,
    )

    all_true: List[str] = []
    all_pred: List[str] = []
    all_paths: List[str] = []
    all_probs: List[np.ndarray] = []

    t0 = time.time()
    n_done = 0
    for waves, labels, paths in loader:
        probs = predict_batch(loaded, waves, device)  # (B, C)
        pred_ids = probs.argmax(axis=-1)
        for i, pid in enumerate(pred_ids):
            all_true.append(labels[i])
            all_pred.append(loaded.id2label[int(pid)])
            all_paths.append(paths[i])
        all_probs.append(probs)
        n_done += len(waves)
        if n_done % (batch_size * 20) == 0 or n_done == len(dataset):
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            print(
                f"[{name}]   {n_done}/{len(dataset)}  "
                f"({rate:.1f} esantioane/s, elapsed {elapsed:.1f}s)",
                flush=True,
            )

    all_probs_arr = np.concatenate(all_probs, axis=0)

    # --- Metrici ---
    labels_sorted = sorted(model_labels)
    acc = accuracy_score(all_true, all_pred)
    f1_macro = f1_score(all_true, all_pred, labels=labels_sorted, average="macro", zero_division=0)
    f1_weighted = f1_score(all_true, all_pred, labels=labels_sorted, average="weighted", zero_division=0)
    report = classification_report(
        all_true, all_pred,
        labels=labels_sorted,
        zero_division=0,
        digits=4,
    )
    cm = confusion_matrix(all_true, all_pred, labels=labels_sorted)

    print(f"\n[{name}] Accuracy       : {acc:.4f}")
    print(f"[{name}] F1 (macro)      : {f1_macro:.4f}")
    print(f"[{name}] F1 (weighted)   : {f1_weighted:.4f}")
    print(f"\n[{name}] Classification report:\n{report}")
    print(f"[{name}] Confusion matrix (etichete: {labels_sorted}):")
    print(cm)

    # --- Salvare pe disc ---
    ds_dir = out_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    # CSV cu predictii per fisier
    import csv
    with open(ds_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["path", "true", "pred"] + [f"prob_{l}" for l in labels_sorted]
        writer.writerow(header)
        # Remapam coloanele de probabilitati in ordinea lui labels_sorted
        col_order = [loaded.label2id[l] for l in labels_sorted]
        for i, p in enumerate(all_paths):
            row = [p, all_true[i], all_pred[i]] + [
                f"{all_probs_arr[i, c]:.6f}" for c in col_order
            ]
            writer.writerow(row)

    # Raport + matrice
    with open(ds_dir / "classification_report.txt", "w") as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"Model:   {MODEL_ID}\n")
        f.write(f"N evaluate: {len(samples)}  (din {len(samples_all)} totale)\n")
        f.write(f"Emotii excluse: {dict(skipped_by_label)}\n\n")
        f.write(f"Accuracy     : {acc:.4f}\n")
        f.write(f"F1 macro     : {f1_macro:.4f}\n")
        f.write(f"F1 weighted  : {f1_weighted:.4f}\n\n")
        f.write(report + "\n\n")
        f.write(f"Confusion matrix (linii = adevar, coloane = predictie)\n")
        f.write("Ordinea etichetelor: " + ", ".join(labels_sorted) + "\n")
        np.savetxt(f, cm, fmt="%d")

    np.save(ds_dir / "confusion_matrix.npy", cm)

    return {
        "name": name,
        "n_total_found": len(samples_all),
        "n_evaluated": len(samples),
        "n_skipped": skipped,
        "skipped_by_label": dict(skipped_by_label),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "label_order": labels_sorted,
        "confusion_matrix": cm.tolist(),
        "true_distribution": dict(label_counts),
    }


# -------------------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluare Wav2Vec2 (Yassmen) pe CREMA-D / RAVDESS / SAVEE / TESS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--crema-d", type=str, default=None,
                   help="Calea catre CREMA-D/AudioWAV")
    p.add_argument("--ravdess", type=str, default=None,
                   help="Calea catre radacina RAVDESS (se cauta recursiv *.wav)")
    p.add_argument("--savee",   type=str, default=None,
                   help="Calea catre SAVEE/AudioData (se cauta recursiv *.wav)")
    p.add_argument("--tess",    type=str, default=None,
                   help="Calea catre TESS (se cauta recursiv *.wav)")
    p.add_argument("--out-dir", type=str, default="./rezultate_wav2vec2_yassmen",
                   help="Director de iesire pentru rapoarte si CSV-uri")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-seconds", type=float, default=8.0,
                   help="Durata maxima per fisier (se taie portiunea centrala)")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu. Daca nu e setat se alege automat.")
    p.add_argument("--limit", type=int, default=None,
                   help="(debug) Limita maxima de fisiere per dataset.")
    return p


def main():
    args = build_argparser().parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[ENV] Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[ENV] GPU:    {torch.cuda.get_device_name(0)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_yassmen_model(device)

    # Corelam argumentele cu parserele
    dataset_roots: Dict[str, Optional[str]] = {
        "CREMA-D": args.crema_d,
        "RAVDESS": args.ravdess,
        "SAVEE":   args.savee,
        "TESS":    args.tess,
    }

    all_results: Dict[str, Dict] = {}
    for name, root in dataset_roots.items():
        if root is None:
            print(f"\n[{name}] --  cale ne-specificata, sar peste.", flush=True)
            continue
        root_path = Path(root)
        if not root_path.exists():
            print(f"\n[{name}] --  calea nu exista: {root_path}, sar peste.", flush=True)
            continue

        parser = PARSERS[name]
        samples = parser(root_path)
        if args.limit:
            samples = samples[: args.limit]

        res = evaluate_dataset(
            name=name,
            samples_all=samples,
            loaded=loaded,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_seconds=args.max_seconds,
            out_dir=out_dir,
        )
        all_results[name] = res

    # Sumar final
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "target_sr": TARGET_SR,
                "canonical_emotions": CANONICAL_EMOTIONS,
                "results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("SUMAR FINAL".center(70))
    print("=" * 70)
    if not all_results:
        print("Nimic evaluat.")
    else:
        print(f"{'Dataset':<12}{'N eval':>10}{'Accuracy':>12}{'F1 macro':>12}{'F1 weight':>12}")
        print("-" * 58)
        for name, r in all_results.items():
            if r.get("n_evaluated", 0) == 0:
                print(f"{name:<12}{'---':>10}{'---':>12}{'---':>12}{'---':>12}")
                continue
            print(
                f"{name:<12}"
                f"{r['n_evaluated']:>10d}"
                f"{r['accuracy']:>12.4f}"
                f"{r['f1_macro']:>12.4f}"
                f"{r['f1_weighted']:>12.4f}"
            )
    print(f"\nRapoarte complete salvate in: {out_dir.resolve()}")
    print(f"Sumar JSON:                    {summary_path.resolve()}")


if __name__ == "__main__":
    main()
