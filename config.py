"""
config.py — Configurare fidela articolului MELECON 2026
"A Multimodal System for Emotion Recognition"

Dataset:  DOAR CREMA-D (fara clasa 'sad' → 5 emotii)
Video:    ViViT (Video Vision Transformer) — 64 cadre, 16 layere, batch 8, LR 1e-5
Audio:    Transformer pe MFCC + Delta + DeltaDelta — 8 layere, 8 heads, LR 1e-6
Fusion:   Weighted Late Fusion cu ponderi learnable
Split:    70% train / 20% val / 10% test (per actor)
"""
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# PATHS — Adapteaza BASE_DIR la calea ta home
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path("/export/home/acs/stud/f/flavius.rau")

# CREMA-D directories
CREMA_DIR       = BASE_DIR / "crema"
CREMA_VIDEO_DIR = CREMA_DIR / "VideoFlash"   # Fisiere .flv
CREMA_AUDIO_DIR = CREMA_DIR / "AudioWAV"     # Fisiere .wav

# Proiect output
PROJECT_DIR = BASE_DIR / "multimodal_emotion"
MODELS_DIR  = PROJECT_DIR / "checkpoints"
LOGS_DIR    = PROJECT_DIR / "logs"
RESULTS_DIR = PROJECT_DIR / "results"

for d in [PROJECT_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6 CLASE DE EMOTII (Tabelul din articol)
# ══════════════════════════════════════════════════════════════════════════════
EMOTION_CLASSES = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
NUM_CLASSES = len(EMOTION_CLASSES)

# Mapare coduri CREMA-D → clase
# Filename format: {ActorID}_{Sentence}_{Emotion}_{Intensity}.flv/.wav
# Ex: 1042_ITH_ANG_XX.flv
# Nota: SAD exclus intentionat
CREMA_EMOTION_MAP = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad',
}

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO — ViViT (Video Vision Transformer)
# Cele mai bune hiperparametri din Tabelul I al articolului:
#   LR=1e-5, NF=64, HL=16, Batch=8 → Acc=86.02%, F1=85.73%
# ══════════════════════════════════════════════════════════════════════════════
VIDEO_NUM_FRAMES    = 64       # Cadre per clip (uniform sampling)
VIDEO_CROP_SIZE     = 360      # Center crop la 360x360 (din 480x360)
VIDEO_RESIZE        = 224      # Resize final la 224x224
VIDEO_NUM_LAYERS    = 16       # Transformer encoder layers (HL)
VIDEO_HIDDEN_SIZE   = 768      # ViViT-Base hidden dimension
VIDEO_NUM_HEADS     = 12       # Attention heads
VIDEO_TUBELET_SIZE  = [2, 16, 16]  # Temporal x Spatial x Spatial
VIDEO_BATCH_SIZE    = 8        # Batch size (din articol)
VIDEO_LR            = 1e-5     # Learning rate (din articol)
VIDEO_DROPOUT       = 0.2      # Dropout probability (din articol)
VIDEO_WEIGHT_DECAY  = 0.01
VIDEO_EPOCHS        = 50
VIDEO_EMBED_DIM     = 256      # Dimensiune embedding pentru fuziune

# ══════════════════════════════════════════════════════════════════════════════
# AUDIO — Transformer pe MFCC + Δ + ΔΔ
# Cele mai bune hiperparametri din Tabelul II:
#   LR=1e-6, AT=8, HL=8, Batch=16 → Acc=81.98%, F1=81.23%
# ══════════════════════════════════════════════════════════════════════════════
AUDIO_SR            = 22050    # Sample rate
AUDIO_DURATION      = 3.0     # Durata clipurilor (2-3 secunde)
AUDIO_N_MFCC        = 40      # Numar coeficienti MFCC
AUDIO_N_FFT         = 2048
AUDIO_HOP_LENGTH    = 512

AUDIO_D_MODEL       = 128     # Dimensiune embedding transformer
AUDIO_NHEAD         = 8       # Attention heads (AT=8)
AUDIO_NUM_LAYERS    = 8       # Transformer encoder layers (HL=8)
AUDIO_DIM_FF        = 512     # Feed-forward dimension
AUDIO_DROPOUT       = 0.2     # Dropout (din articol)
AUDIO_BATCH_SIZE    = 16      # Batch size (din articol)
AUDIO_LR            = 1e-6    # Learning rate (din articol)
AUDIO_WEIGHT_DECAY  = 1e-4
AUDIO_EPOCHS        = 60
AUDIO_EMBED_DIM     = 256     # Dimensiune embedding pentru fuziune

# ══════════════════════════════════════════════════════════════════════════════
# FUSION — Weighted Late Fusion (Sectiunea III.A din articol)
# ══════════════════════════════════════════════════════════════════════════════
FUSION_BATCH_SIZE   = 16
FUSION_EPOCHS       = 40
FUSION_LR           = 5e-4
FUSION_WEIGHT_DECAY = 1e-4
FUSION_DROPOUT      = 0.3
FUSION_HIDDEN_DIM   = 256

# ══════════════════════════════════════════════════════════════════════════════
# GENERAL
# ══════════════════════════════════════════════════════════════════════════════
RANDOM_SEED  = 42
NUM_WORKERS  = 4
TEST_SIZE    = 0.10          # 10% din total → test
VAL_SIZE     = 0.20 / 0.90  # 20% din total → val (calculat din 90% ramas dupa test)


VIDEO_EPOCHS_WARM = 5
VIDEO_EPOCHS_FINE = 30
VIDEO_LR_HEAD = 1e-3
VIDEO_LR_BACKBONE = 1e-5
VIDEO_WEIGHT_DECAY = 0.05

