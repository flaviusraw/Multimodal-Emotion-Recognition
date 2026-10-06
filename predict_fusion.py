"""
predict_fusion.py — Late fusion intre ViViT (video) si HuBERT (audio)
Medie simpla a probabilitatilor celor doua modele.

Extrage audio din videoclip cu ffmpeg.

Utilizare:
  python predict_fusion.py --video clip.mp4
  python predict_fusion.py --video clip.mp4 --video-weight 0.6 --audio-weight 0.4
  python predict_fusion.py --video clip.mp4 --show-individual
"""
import argparse
import subprocess
import tempfile
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter

# ── HuggingFace imports pentru HuBERT ────────────────────────────────────────
from transformers import AutoFeatureExtractor
from train_wav2vec2_cremad import HubertForSpeechClassification
import librosa

# ── Proiect imports ───────────────────────────────────────────────────────────
from config import (
    EMOTION_CLASSES, NUM_CLASSES,
    VIDEO_NUM_FRAMES, VIDEO_RESIZE, VIDEO_EMBED_DIM, VIDEO_DROPOUT,
)
from models import VideoViViT

# ── Cai modele ────────────────────────────────────────────────────────────────
MODELS_DIR       = Path("checkpoints")
HUBERT_MODEL_DIR = Path("/export/home/acs/stud/f/flavius.rau/multimodal_emotion/runs/chinese_hubert_samespeaker/best_model")   # folderul cu config.json + model.safetensors

# Mapare clase HuBERT → EMOTION_CLASSES
# HuBERT: ["angry","disgust","fear","happy","neutral","sad"]
# Proiect: ["angry","disgust","fearful","happy","neutral","sad"]
HUBERT_TO_PROJECT = {
    'angry':   'angry',
    'disgust': 'disgust',
    'fear':    'fearful',
    'happy':   'happy',
    'neutral': 'neutral',
    'sad':     'sad',
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

EMOTION_COLORS = {
    'angry':   (0,   0,   220),
    'disgust': (0,   140, 0  ),
    'fearful': (180, 0,   180),
    'happy':   (0,   200, 200),
    'neutral': (180, 180, 180),
    'sad':     (220, 100, 0  ),
}


# ════════════════════════════════════════════════════════════════════════════
# VIDEO
# ════════════════════════════════════════════════════════════════════════════

def frames_to_tensor(frames_rgb, device):
    """Center crop 360x360 → resize 224x224 (consistent cu antrenarea)."""
    processed = []
    for frame in frames_rgb:
        h, w = frame.shape[:2]
        side = min(h, w, 360)
        ys = (h - side) // 2
        xs = (w - side) // 2
        crop = frame[ys:ys+side, xs:xs+side]
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        resized = resized.astype(np.float32) / 255.0
        resized = (resized - MEAN) / STD
        processed.append(resized)

    arr    = np.stack(processed, axis=0)
    tensor = torch.tensor(arr).permute(3, 0, 1, 2).unsqueeze(0)
    return tensor.to(device)


def load_video_model(device):
    model = VideoViViT(
        num_classes=NUM_CLASSES,
        embed_dim=VIDEO_EMBED_DIM,
        dropout=VIDEO_DROPOUT,
    ).to(device)

    ckpt_path = MODELS_DIR / f"video_vit_best_{NUM_CLASSES}cls.pt"
    if not ckpt_path.exists():
        ckpt_path = MODELS_DIR / "video_vit_best.pt"

    print(f"  [Video] Incarcare: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  [Video] Epoch: {ckpt['epoch']} | Val acc: {ckpt['val_acc']:.4f}")
    return model


@torch.no_grad()
def predict_video_segment(frames_rgb, video_model, device):
    """Returneaza probs array aliniat la EMOTION_CLASSES."""
    indices = np.linspace(0, len(frames_rgb)-1, VIDEO_NUM_FRAMES, dtype=int)
    sampled = [frames_rgb[i] for i in indices]
    tensor  = frames_to_tensor(sampled, device)
    logits  = video_model(tensor)
    probs   = F.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs  # aliniat la EMOTION_CLASSES din config


# ════════════════════════════════════════════════════════════════════════════
# AUDIO — HuBERT
# ════════════════════════════════════════════════════════════════════════════

def load_hubert_model(device):
    print(f"  [Audio] Incarcare HuBERT din: {HUBERT_MODEL_DIR}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(HUBERT_MODEL_DIR))
    model = HubertForSpeechClassification.from_pretrained(str(HUBERT_MODEL_DIR))
    model.to(device)
    model.eval()

    # Citeste ordinea claselor din config HuBERT
    hubert_labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    print(f"  [Audio] Clase HuBERT: {hubert_labels}")
    return feature_extractor, model, hubert_labels


def extract_audio_from_video(video_path, sr=16000):
    """Extrage audio din videoclip cu ffmpeg, returneaza numpy array mono."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-ac', '1',           # mono
        '-ar', str(sr),       # sample rate
        '-vn',                # fara video
        '-f', 'wav',
        tmp_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  [WARN] ffmpeg eroare: {result.stderr.decode()[:200]}")
        return None, sr

    audio, _ = librosa.load(tmp_path, sr=sr, mono=True)
    os.unlink(tmp_path)
    return audio, sr


@torch.no_grad()
def predict_audio_segment(audio_samples, sr, feature_extractor, hubert_model,
                           hubert_labels, device):
    """
    Returneaza probs array aliniat la EMOTION_CLASSES (nu la HuBERT labels).
    """
    if audio_samples is None or len(audio_samples) == 0:
        return np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES

    inputs = feature_extractor(
        audio_samples, sampling_rate=sr,
        return_tensors='pt', padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = hubert_model(**inputs).logits
    hubert_probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # Realiniaza la EMOTION_CLASSES
    project_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, hlabel in enumerate(hubert_labels):
        proj_label = HUBERT_TO_PROJECT.get(hlabel)
        if proj_label and proj_label in EMOTION_CLASSES:
            j = EMOTION_CLASSES.index(proj_label)
            project_probs[j] = hubert_probs[i]

    # Normalizeaza
    total = project_probs.sum()
    if total > 0:
        project_probs /= total

    return project_probs


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Late fusion ViViT + HuBERT pe videoclip'
    )
    parser.add_argument('--video',         type=str, required=True)
    parser.add_argument('--device',        type=str, default=None)
    parser.add_argument('--video-weight',  type=float, default=0.5,
                        help='Ponderea modelului video (default: 0.5)')
    parser.add_argument('--audio-weight',  type=float, default=0.5,
                        help='Ponderea modelului audio (default: 0.5)')
    parser.add_argument('--wav', type=str, default=None, help='Cale fisier .wav (alternativa la ffmpeg)')
    parser.add_argument('--segment-sec',   type=float, default=2.5)
    parser.add_argument('--show-individual', action='store_true',
                        help='Afiseaza si predictiile individuale video/audio')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    vw = args.video_weight / (args.video_weight + args.audio_weight)
    aw = args.audio_weight / (args.video_weight + args.audio_weight)

    print(f"\n{'='*55}")
    print(f"  LATE FUSION — ViViT ({vw*100:.0f}%) + HuBERT ({aw*100:.0f}%)")
    print(f"  Device: {device}")
    print(f"{'='*55}")

    # Incarcare modele
    print("\n[INFO] Incarcare modele...")
    video_model = load_video_model(device)
    feature_extractor, hubert_model, hubert_labels = load_hubert_model(device)

    # Extrage audio din videoclip
    print(f"\n[INFO] Extragere audio din: {args.video}")
    if args.wav:
        import librosa as _lib
        audio_samples, sr = _lib.load(args.wav, sr=16000, mono=True)
    else:
        audio_samples, sr = extract_audio_from_video(args.video)
    if audio_samples is not None:
        print(f"  Audio: {len(audio_samples)/sr:.2f}s @ {sr}Hz")
    else:
        print("  [WARN] Audio indisponibil — se foloseste doar video")

    # Citeste frame-urile video
    cap        = cv2.VideoCapture(args.video)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total_fr / fps
    seg_frames = int(fps * args.segment_sec)
    n_segments = max(1, int(duration / args.segment_sec))

    all_frames = []
    ret, frame = cap.read()
    while ret:
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ret, frame = cap.read()
    cap.release()

    print(f"\n[INFO] Video: {duration:.1f}s | {n_segments} segmente")
    print(f"{'='*55}")

    seg_results = []

    with torch.no_grad():
        for seg_idx in range(n_segments):
            v_start = seg_idx * seg_frames
            v_end   = min(v_start + seg_frames, len(all_frames))
            seg     = all_frames[v_start:v_end]
            if len(seg) < 4:
                continue

            # Audio pentru acest segment
            a_start = int(v_start / fps * sr)
            a_end   = int(v_end   / fps * sr)
            seg_audio = audio_samples[a_start:a_end] \
                        if audio_samples is not None else None

            # Predictii individuale
            video_probs = predict_video_segment(seg, video_model, device)
            audio_probs = predict_audio_segment(
                seg_audio, sr, feature_extractor, hubert_model, hubert_labels, device
            )

            # Fuziune — medie ponderata
            fusion_probs = vw * video_probs + aw * audio_probs
            fusion_emotion = EMOTION_CLASSES[np.argmax(fusion_probs)]

            seg_results.append((fusion_emotion, fusion_probs, video_probs, audio_probs))

            v_em = EMOTION_CLASSES[np.argmax(video_probs)]
            a_em = EMOTION_CLASSES[np.argmax(audio_probs)]

            if args.show_individual:
                print(f"  Seg {seg_idx+1:2d} | "
                      f"Video: {v_em:8s}({video_probs.max()*100:.0f}%) | "
                      f"Audio: {a_em:8s}({audio_probs.max()*100:.0f}%) | "
                      f"Fusion: {fusion_emotion.upper():8s}({fusion_probs.max()*100:.0f}%)")
            else:
                print(f"  Segment {seg_idx+1:2d}/{n_segments} → "
                      f"{fusion_emotion.upper():8s} ({fusion_probs.max()*100:.1f}%)")

    if not seg_results:
        print("[WARN] Niciun segment procesat.")
        return

    # ── Rezultat final ────────────────────────────────────────────────────────
    mean_fusion = np.mean([p for _, p, _, _ in seg_results], axis=0)
    mean_video  = np.mean([vp for _, _, vp, _ in seg_results], axis=0)
    mean_audio  = np.mean([ap for _, _, _, ap in seg_results], axis=0)

    final_fusion  = EMOTION_CLASSES[np.argmax(mean_fusion)]
    final_video   = EMOTION_CLASSES[np.argmax(mean_video)]
    final_audio   = EMOTION_CLASSES[np.argmax(mean_audio)]
    votes         = [e for e, _, _, _ in seg_results]
    majority, cnt = Counter(votes).most_common(1)[0]

    print(f"\n{'='*55}")
    print(f"  REZULTAT FINAL")
    print(f"  Fuziune:  {final_fusion.upper():10s} ({mean_fusion.max()*100:.1f}%)")
    print(f"  Video:    {final_video.upper():10s} ({mean_video.max()*100:.1f}%)")
    print(f"  Audio:    {final_audio.upper():10s} ({mean_audio.max()*100:.1f}%)")
    print(f"  Majoritar:{majority.upper():10s} ({cnt}/{len(votes)} segmente)")
    print(f"\n  Probabilitati medii fuziune:")
    for em, prob in sorted(zip(EMOTION_CLASSES, mean_fusion), key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"    {em:8s} {prob*100:5.1f}%  {bar}")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
