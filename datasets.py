"""
datasets.py — Dataset-uri fidele articolului
  VideoClipDataset: citeste fisiere video (.flv/.mp4), extrage 64 cadre
                    uniform sampling, center crop 360x360, resize 224x224
                    → tensor (C, T, H, W) = (3, 64, 224, 224)
  AudioMFCCDataset: MFCC + Δ + ΔΔ ca secventa temporala
                    → tensor (T, 120)
"""
import cv2
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset

from config import *

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_CLASSES)}


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO DATASET — Citeste video, extrage 64 cadre, center crop, resize
# ══════════════════════════════════════════════════════════════════════════════

def _read_video_frames(video_path, num_frames=VIDEO_NUM_FRAMES):
    """
    Citeste un fisier video si returneaza toate cadrele ca lista de numpy arrays.
    Foloseste OpenCV (suporta .flv, .mp4, .avi).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Nu pot deschide video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise IOError(f"Video gol: {video_path}")

    return frames


def _uniform_sample_frames(frames, num_frames=VIDEO_NUM_FRAMES):
    """
    Esantioneaza exact `num_frames` cadre distribuite uniform pe toata durata.
    Din articol: "frames were selected using a uniform sampling strategy,
    ensuring that the chosen frames were evenly spaced across the entire video"
    """
    total = len(frames)

    if total >= num_frames:
        # Selecteaza uniform
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        # Daca videoul e prea scurt, repeta ultimul cadru
        indices = list(range(total))
        while len(indices) < num_frames:
            indices.append(total - 1)
        indices = np.array(indices[:num_frames])

    return [frames[i] for i in indices]


def _center_crop_horizontal(frame, crop_size=VIDEO_CROP_SIZE):
    """
    Decupare orizontala centrala pentru a obtine o regiune patrata.
    Din articol: "frames were cropped horizontally to obtain a 360x360 square
    region, ensuring that the cropping was symmetric on both left and right sides"

    CREMA-D: 480x360 pixels → center crop 360x360
    """
    h, w = frame.shape[:2]

    # Calculeaza crop-ul patrat centrat
    crop_h = min(h, crop_size)
    crop_w = min(w, crop_size)
    side = min(crop_h, crop_w)

    y_start = (h - side) // 2
    x_start = (w - side) // 2

    cropped = frame[y_start:y_start + side, x_start:x_start + side]
    return cropped


def _resize_frame(frame, size=VIDEO_RESIZE):
    """Redimensioneaza cadrul la size x size (224x224)."""
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)


# Incarca detectorul Haar Cascade o singura data la nivel de modul (nu per-frame)
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def _detect_face_and_resize(frame_rgb, out_size=VIDEO_RESIZE, padding=0.20):
    """
    Detecteaza fata pe frame-ul original (480x360 RGB) cu Haar Cascade
    si taie un patrat de out_size x out_size (224x224) centrat pe fata,
    direct din frame — fara resize, fara distorsiune.
    Daca fata e prea aproape de margini, se completeaza cu negru (zero-padding).
    Fallback: center crop 224x224 din mijlocul frame-ului daca nu se gaseste fata.

    Args:
        frame_rgb: numpy array (H, W, 3) uint8, RGB
        out_size:  dimensiunea ferestrei patrate in jurul fetei (default 224)
        padding:   proportie extra in jurul bbox fetei (default 20%)
    Returns:
        numpy array (out_size, out_size, 3) uint8, RGB — intotdeauna 224x224
    """
    h, w = frame_rgb.shape[:2]
    half = out_size // 2

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        # Fallback: crop 224x224 centrat pe mijlocul frame-ului
        cx, cy = w // 2, h // 2
    else:
        # Cea mai mare fata detectata
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        # Centrul fetei (cu padding inclus)
        pad_x = int(fw * padding)
        pad_y = int(fh * padding)
        cx = x + fw // 2
        cy = y + fh // 2
        # Extinde fereastra sa includa padding-ul
        half = max(half, (fw // 2 + pad_x), (fh // 2 + pad_y))
        half = min(half, out_size)  # nu depasi 224

    # Coordonatele ferestrei 224x224 centrate pe (cx, cy)
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    # Daca fereastra iese din frame, adauga zero-padding
    pad_top    = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left   = max(0, -x1)
    pad_right  = max(0, x2 - w)

    # Clip la marginile frame-ului
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)

    crop = frame_rgb[y1c:y2c, x1c:x2c]

    # Aplica zero-padding daca e necesar
    if pad_top or pad_bottom or pad_left or pad_right:
        crop = np.pad(crop,
                      ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                      mode='constant', constant_values=0)

    # Asigura dimensiunea exacta out_size x out_size
    if crop.shape[0] != out_size or crop.shape[1] != out_size:
        crop = cv2.resize(crop, (out_size, out_size),
                          interpolation=cv2.INTER_LINEAR)

    return crop


def _preprocess_video(video_path, num_frames=VIDEO_NUM_FRAMES,
                       crop_size=VIDEO_CROP_SIZE, resize=VIDEO_RESIZE):
    """
    Pipeline complet de preprocesare video:
      1. Citeste toate cadrele din fisierul video
      2. Uniform sampling → 64 cadre
      3. Haar Cascade pe frame original 480x360 → fereastra 224x224 centrata pe fata
         (fara resize, fara distorsiune — zero-padding la margini daca e necesar)
         (fallback: crop 224x224 centrat pe mijlocul frame-ului)
      4. Normalizare [0, 1]
    Returneaza: numpy array (num_frames, H, W, 3) float32
    """
    # 1. Citeste cadrele
    all_frames = _read_video_frames(video_path)

    # 2. Uniform sampling
    sampled = _uniform_sample_frames(all_frames, num_frames)

    # 3-4. Haar Cascade → crop fata → 224x224 direct (sau frame intreg fallback)
    processed = []
    for frame in sampled:
        out = _detect_face_and_resize(frame, out_size=resize)  # 224x224 garantat
        processed.append(out)

    # Stack si normalizare
    video_array = np.stack(processed, axis=0).astype(np.float32) / 255.0

    return video_array  # (T, H, W, 3)


class VideoClipDataset(Dataset):
    """
    Dataset video CREMA-D — citeste direct fisierele .flv/.mp4.

    Fiecare sample returneaza:
      video: tensor (C, T, H, W) = (3, 64, 224, 224)
      label: int (0-5)

    Forma (C, T, H, W) e standard pentru modele video (PyTorchVideo, ViViT).
    """

    def __init__(self, csv_path, num_frames=VIDEO_NUM_FRAMES,
                 augment=False):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['emotion'].isin(EMOTION_CLASSES)].reset_index(drop=True)
        self.num_frames = num_frames
        self.augment = augment

        # ImageNet normalization stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _normalize(self, video):
        """Aplica ImageNet normalizare: (T, H, W, 3) float32 in [0,1]"""
        return (video - self.mean) / self.std

    def _augment_video(self, video):
        """
        Augmentare simpla pe secventa video.
        video: (T, H, W, 3) float32 [0, 1]
        """
        # Random horizontal flip (acelasi flip pt toate cadrele)
        if np.random.random() < 0.5:
            video = video[:, :, ::-1, :].copy()

        # Random brightness
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.85, 1.15)
            video = np.clip(video * factor, 0, 1)

        # Random contrast
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.85, 1.15)
            mean = video.mean(axis=(1, 2, 3), keepdims=True)
            video = np.clip((video - mean) * factor + mean, 0, 1)

        return video

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            # Preprocesare video: citire → uniform sampling → crop → resize
            video = _preprocess_video(
                row['video_path'],
                num_frames=self.num_frames,
            )
            # video: (T, H, W, 3) float32 [0, 1]

            # Augmentare (doar la train)
            if self.augment:
                video = self._augment_video(video)

            # Normalizare ImageNet
            video = self._normalize(video)

            # Converteste la (C, T, H, W) — format standard video
            # (T, H, W, C) → (C, T, H, W)
            video_tensor = torch.tensor(video, dtype=torch.float32).permute(3, 0, 1, 2)

        except Exception as e:
            # Fallback: tensor de zerouri (semnaleaza eroare silentios)
            print(f"  WARNING: Nu pot citi video {row['video_path']}: {e}")
            video_tensor = torch.zeros(3, self.num_frames, VIDEO_RESIZE, VIDEO_RESIZE,
                                        dtype=torch.float32)

        label = EMOTION_TO_IDX[row['emotion']]
        return video_tensor, label


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO DATASET — MFCC + Delta + DeltaDelta (Sectiunea III.A din articol)
# ══════════════════════════════════════════════════════════════════════════════

def extract_mfcc_features(file_path, sr=AUDIO_SR, duration=AUDIO_DURATION,
                           n_mfcc=AUDIO_N_MFCC, n_fft=AUDIO_N_FFT,
                           hop_length=AUDIO_HOP_LENGTH):
    """
    Extrage MFCC + Delta(Δ) + DeltaDelta(ΔΔ) dintr-un fisier audio.

    Din articol (Sectiunea III.A):
      "we extracted delta (Δ) and delta-delta (ΔΔ) features, which represent
       the first- and second-order temporal derivatives of the spectral
       coefficients. The three components are stacked along the feature dimension,
       producing a sequence of frame-level vectors."

    Returneaza: numpy array (time_steps, n_mfcc * 3) = (T, 120)
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=25)

        # Pad/truncate la durata fixa
        n_samples = int(sr * duration)
        if len(y) < n_samples:
            y = np.pad(y, (0, n_samples - len(y)), mode='constant')
        else:
            y = y[:n_samples]

        # Normalizare audio
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # Log-Mel / MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop_length)
        # Delta si DeltaDelta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack: (3 * n_mfcc, T) → transpus (T, 3 * n_mfcc)
        features = np.vstack([mfcc, delta, delta2]).T

        return features.astype(np.float32)

    except Exception as e:
        print(f"  WARNING: Nu pot procesa audio {file_path}: {e}")
        return None


class AudioMFCCDataset(Dataset):
    """
    Dataset audio CREMA-D — MFCC + Δ + ΔΔ ca secventa temporala.

    Din articol: "Instead of treating the spectrogram as an image, we structured
    the features as a sequence of tabular vectors, each corresponding to a time
    frame."

    Fiecare sample returneaza:
      features: tensor (T, feature_dim) = (max_len, 120)
      label: int (0-5)
    """

    def __init__(self, csv_path, max_len=None, augment=False):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['emotion'].isin(EMOTION_CLASSES)].reset_index(drop=True)
        self.augment = augment

        # Calculeaza max_len din parametri audio
        if max_len is None:
            n_samples = int(AUDIO_SR * AUDIO_DURATION)
            self.max_len = 1 + n_samples // AUDIO_HOP_LENGTH
        else:
            self.max_len = max_len

        self.feature_dim = AUDIO_N_MFCC * 3  # MFCC + Delta + DeltaDelta = 120

    def __len__(self):
        return len(self.df)

    def _augment_features(self, features):
        """SpecAugment-like augmentare pe features."""
        features = features.copy()

        # Time masking
        if np.random.random() < 0.5:
            t = features.shape[0]
            mask_len = min(int(t * 0.15), 10)
            if mask_len > 0 and t > mask_len:
                start = np.random.randint(0, t - mask_len)
                features[start:start + mask_len, :] = 0

        # Feature masking
        if np.random.random() < 0.5:
            f = features.shape[1]
            mask_len = min(int(f * 0.1), 8)
            if mask_len > 0 and f > mask_len:
                start = np.random.randint(0, f - mask_len)
                features[:, start:start + mask_len] = 0

        # Gaussian noise
        if np.random.random() < 0.3:
            features += np.random.randn(*features.shape).astype(np.float32) * 0.01

        return features

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = extract_mfcc_features(row['audio_path'])

        if features is None:
            features = np.zeros((self.max_len, self.feature_dim), dtype=np.float32)

        if self.augment:
            features = self._augment_features(features)

        # Pad/truncate la dimensiune fixa
        T, F = features.shape
        if T < self.max_len:
            features = np.vstack([features,
                                  np.zeros((self.max_len - T, F), dtype=np.float32)])
        else:
            features = features[:self.max_len]

        # Normalizare per-sample
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        label = EMOTION_TO_IDX[row['emotion']]
        return torch.tensor(features, dtype=torch.float32), label


# ══════════════════════════════════════════════════════════════════════════════
# FUSION DATASET (pentru embedding-uri pre-extrase)
# ══════════════════════════════════════════════════════════════════════════════

class FusionDataset(Dataset):
    """Dataset cu embedding-uri pre-calculate (video + audio)."""

    def __init__(self, video_emb, audio_emb, labels):
        self.video_emb = torch.tensor(video_emb, dtype=torch.float32)
        self.audio_emb = torch.tensor(audio_emb, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.video_emb[idx], self.audio_emb[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO HAAR DATASET — Citeste frame-uri JPEG pre-extrase cu Haar Cascade
# Mult mai rapid decat VideoClipDataset (nu citeste .flv la runtime)
# ══════════════════════════════════════════════════════════════════════════════

class VideoHaarDataset(Dataset):
    """
    Dataset video care citeste frame-uri JPEG pre-extrase cu Haar Cascade
    din folderul haar_frames/.

    Structura asteptata:
      haar_frames_dir/
        1001_IEO_ANG_LO/
          frame_00.jpg ... frame_63.jpg
        1001_IEO_DIS_LO/
          ...

    Fiecare sample returneaza:
      video: tensor (C, T, H, W) = (3, 64, 224, 224)
      label: int (0-5)
    """

    def __init__(self, csv_path, haar_frames_dir, num_frames=VIDEO_NUM_FRAMES,
                 augment=False):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['emotion'].isin(EMOTION_CLASSES)].reset_index(drop=True)
        self.haar_dir  = Path(haar_frames_dir)
        self.num_frames = num_frames
        self.augment   = augment

        # ImageNet normalizare
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Verifica cate clipuri au folderul haar disponibil
        available = sum(
            1 for clip_id in self.df['clip_id']
            if (self.haar_dir / clip_id).exists()
        )
        print(f"  [HaarDataset] {available}/{len(self.df)} clipuri cu haar frames disponibile")

    def __len__(self):
        return len(self.df)

    def _normalize(self, video):
        return (video - self.mean) / self.std

    def _augment_video(self, video):
        if np.random.random() < 0.5:
            video = video[:, :, ::-1, :].copy()
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.85, 1.15)
            video = np.clip(video * factor, 0, 1)
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.85, 1.15)
            mean = video.mean(axis=(1, 2, 3), keepdims=True)
            video = np.clip((video - mean) * factor + mean, 0, 1)
        return video

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        clip_id  = row['clip_id']
        clip_dir = self.haar_dir / clip_id

        try:
            # Citeste frame-urile JPEG sortate
            jpg_paths = sorted(clip_dir.glob("frame_*.jpg"))

            if len(jpg_paths) == 0:
                raise FileNotFoundError(f"Niciun frame in {clip_dir}")

            # Uniform sampling daca numarul de frame-uri difera de cel asteptat
            if len(jpg_paths) == self.num_frames:
                selected = jpg_paths
            else:
                indices  = np.linspace(0, len(jpg_paths)-1, self.num_frames, dtype=int)
                selected = [jpg_paths[i] for i in indices]

            frames = []
            for jpg in selected:
                img_bgr = cv2.imread(str(jpg))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                frames.append(img_rgb)

            video = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)

            if self.augment:
                video = self._augment_video(video)

            video = self._normalize(video)
            video_tensor = torch.tensor(video, dtype=torch.float32).permute(3, 0, 1, 2)

        except Exception as e:
            print(f"  WARNING: {clip_id}: {e}")
            video_tensor = torch.zeros(3, self.num_frames, VIDEO_RESIZE, VIDEO_RESIZE,
                                       dtype=torch.float32)

        label = EMOTION_TO_IDX[row['emotion']]
        return video_tensor, label
