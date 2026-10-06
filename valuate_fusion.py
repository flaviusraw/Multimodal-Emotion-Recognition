"""
evaluate_fusion.py — Evalueaza late fusion ViViT + HuBERT pe test.csv
Afiseaza accuracy, F1 macro, classification report si confusion matrix.

Utilizare:
  python evaluate_fusion.py
  python evaluate_fusion.py --split test    # doar test set (default)
  python evaluate_fusion.py --split all     # toate clipurile
  python evaluate_fusion.py --video-weight 0.6 --audio-weight 0.4
"""
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from transformers import AutoFeatureExtractor
from train_wav2vec2_cremad import HubertForSpeechClassification

from config import (
    EMOTION_CLASSES, NUM_CLASSES, PROJECT_DIR,
    VIDEO_NUM_FRAMES, VIDEO_RESIZE, VIDEO_EMBED_DIM, VIDEO_DROPOUT,
)
from models import VideoViViT

# ── Cai ──────────────────────────────────────────────────────────────────────
MODELS_DIR       = Path("/export/home/acs/stud/f/flavius.rau/multimodal_emotion/checkpoints")
HUBERT_MODEL_DIR = Path("/export/home/acs/stud/f/flavius.rau/multimodal_emotion/runs/chinese_hubert_samespeaker/best_model")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HUBERT_TO_PROJECT = {
    'angry':   'angry',
    'disgust': 'disgust',
    'fear':    'fearful',
    'happy':   'happy',
    'neutral': 'neutral',
    'sad':     'sad',
}


# ── Video ─────────────────────────────────────────────────────────────────────

def load_video_model(device):
    model = VideoViViT(
        num_classes=NUM_CLASSES,
        embed_dim=VIDEO_EMBED_DIM,
        dropout=VIDEO_DROPOUT,
    ).to(device)
    ckpt_path = MODELS_DIR / "video_vit_haar_best_6cls.pt"
    if not ckpt_path.exists():
        ckpt_path = MODELS_DIR / "video_vit_haar_best_6cls.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  [Video] {ckpt_path.name} | epoch={ckpt['epoch']} | val_acc={ckpt['val_acc']:.4f}")
    return model


def predict_video(video_path, model, device):
    cap    = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        return np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES

    indices = np.linspace(0, len(frames)-1, VIDEO_NUM_FRAMES, dtype=int)
    sampled = [frames[i] for i in indices]

    # Detectie fata pe frame-ul din mijloc, aplica acelasi crop la toate
    mid = sampled[len(sampled) // 2]
    h, w = mid.shape[:2]
    half = 112
    gray = cv2.cvtColor(mid, cv2.COLOR_RGB2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        cx, cy = x + fw//2, y + fh//2
    else:
        cx, cy = w//2, h//2
    x1, y1, x2, y2 = cx-half, cy-half, cx+half, cy+half

    processed = []
    for frame in sampled:
        pad_top    = max(0, -y1)
        pad_bottom = max(0, y2 - h)
        pad_left   = max(0, -x1)
        pad_right  = max(0, x2 - w)
        crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if pad_top or pad_bottom or pad_left or pad_right:
            crop = np.pad(crop, ((pad_top,pad_bottom),(pad_left,pad_right),(0,0)),
                          mode="constant", constant_values=0)
        if crop.shape[0] != 224 or crop.shape[1] != 224:
            crop = cv2.resize(crop, (224, 224))
        resized = crop.astype(np.float32) / 255.0
        resized = (resized - MEAN) / STD
        processed.append(resized)

    arr    = np.stack(processed, axis=0)
    tensor = torch.tensor(arr).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
    return F.softmax(logits, dim=-1).cpu().numpy()[0]


# ── Audio ─────────────────────────────────────────────────────────────────────

def load_hubert_model(device):
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(HUBERT_MODEL_DIR))
    model = HubertForSpeechClassification.from_pretrained(str(HUBERT_MODEL_DIR))
    model.to(device).eval()
    hubert_labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    print(f"  [Audio] HuBERT | labels={hubert_labels}")
    return feature_extractor, model, hubert_labels


def predict_audio(audio_path, feature_extractor, hubert_model, hubert_labels, device):
    try:
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    except Exception:
        return np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES

    inputs = feature_extractor(
        audio, sampling_rate=16000, return_tensors='pt', padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = hubert_model(**inputs).logits
    hubert_probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # Realiniaza la EMOTION_CLASSES
    project_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, hlabel in enumerate(hubert_labels):
        proj = HUBERT_TO_PROJECT.get(hlabel)
        if proj and proj in EMOTION_CLASSES:
            project_probs[EMOTION_CLASSES.index(proj)] = hubert_probs[i]

    total = project_probs.sum()
    if total > 0:
        project_probs /= total
    return project_probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',        type=str,   default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Split de evaluat (default: test)')
    parser.add_argument('--device',       type=str,   default=None)
    parser.add_argument('--video-weight', type=float, default=0.5)
    parser.add_argument('--audio-weight', type=float, default=0.5)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    vw = args.video_weight / (args.video_weight + args.audio_weight)
    aw = args.audio_weight / (args.video_weight + args.audio_weight)

    print(f"\n{'='*60}")
    print(f"  EVALUARE FUZIUNE — ViViT ({vw*100:.0f}%) + HuBERT ({aw*100:.0f}%)")
    print(f"  Split: {args.split} | Device: {device}")
    print(f"{'='*60}\n")

    # Incarcare modele
    print("[INFO] Incarcare modele...")
    video_model = load_video_model(device)
    feature_extractor, hubert_model, hubert_labels = load_hubert_model(device)

    # Incarcare CSV
    if args.split == 'all':
        df = pd.read_csv(PROJECT_DIR / "crema_all.csv")
    else:
        df = pd.read_csv(PROJECT_DIR / f"{args.split}.csv")

    df = df[df['emotion'].isin(EMOTION_CLASSES)].reset_index(drop=True)
    print(f"\n[INFO] Clipuri de evaluat: {len(df)}")

    # Evaluare
    y_true, y_pred_fusion, y_pred_video, y_pred_audio = [], [], [], []
    errors = 0

    bar = tqdm(df.iterrows(), total=len(df), desc="Evaluare")
    for _, row in bar:
        try:
            video_probs = predict_video(row['video_path'], video_model, device)
            audio_probs = predict_audio(row['audio_path'],
                                        feature_extractor, hubert_model,
                                        hubert_labels, device)
            fusion_probs = vw * video_probs + aw * audio_probs

            true_idx = EMOTION_CLASSES.index(row['emotion'])
            y_true.append(true_idx)
            y_pred_fusion.append(np.argmax(fusion_probs))
            y_pred_video.append(np.argmax(video_probs))
            y_pred_audio.append(np.argmax(audio_probs))

        except Exception as e:
            errors += 1
            bar.set_postfix(errors=errors)

    # Metrici
    print(f"\n{'='*60}")
    print(f"  REZULTATE ({args.split.upper()} SET — {len(y_true)} clipuri)")
    print(f"{'='*60}\n")

    for name, preds in [('VIDEO only', y_pred_video),
                        ('AUDIO only', y_pred_audio),
                        ('FUSION',     y_pred_fusion)]:
        acc = accuracy_score(y_true, preds)
        f1  = f1_score(y_true, preds, average='macro')
        print(f"  {name:12s} → Accuracy: {acc*100:.2f}% | F1 macro: {f1*100:.2f}%")

    print(f"\n  Classification Report (FUSION):")
    print(classification_report(y_true, y_pred_fusion,
                                target_names=EMOTION_CLASSES, digits=4))

    print(f"  Confusion Matrix (FUSION):")
    cm = confusion_matrix(y_true, y_pred_fusion)
    header = f"{'':10s}" + "".join(f"{e[:6]:>8s}" for e in EMOTION_CLASSES)
    print(f"  {header}")
    for i, row_cm in enumerate(cm):
        row_str = f"  {EMOTION_CLASSES[i]:10s}" + "".join(f"{v:>8d}" for v in row_cm)
        print(row_str)

    if errors > 0:
        print(f"\n  [WARN] {errors} clipuri au generat erori si au fost sarite.")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
