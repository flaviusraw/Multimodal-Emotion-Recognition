"""
plot_confusion_matrices.py — Genereaza matrici de confuzie estetice
pentru modelele video (Haar), audio (HuBERT) si fuziune.

Subsetul de test este echilibrat: numar egal de exemple per clasa (~10-12%).
Se combina test.csv + val.csv pentru a atinge minim 120 exemple/clasa.

Utilizare (pe server):
  python plot_confusion_matrices.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import confusion_matrix
from transformers import AutoFeatureExtractor
import torch.nn as nn
from transformers import HubertPreTrainedModel, HubertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import librosa
import sys
sys.path.insert(0, '/export/home/acs/stud/f/flavius.rau/multimodal_emotion')

from config import EMOTION_CLASSES, NUM_CLASSES, PROJECT_DIR, MODELS_DIR
from models import VideoViViT
from config import VIDEO_NUM_FRAMES, VIDEO_RESIZE, VIDEO_EMBED_DIM, VIDEO_DROPOUT

HUBERT_MODEL_DIR = PROJECT_DIR / 'runs' / 'chinese_hubert_samespeaker' / 'best_model'
RESULTS_DIR = PROJECT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VIDEO_WEIGHT = 0.35
AUDIO_WEIGHT = 0.65

# Etichete scurte pentru matrice
CLASS_LABELS = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad']

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── HuBERT arhitectura custom ─────────────────────────────────────────────────
class HubertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert               = HubertModel(config)
        self.classifier           = HubertClassificationHead(config)
        self._tied_weights_keys   = []
        self.all_tied_weights_keys = {}
        self.init_weights()

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs       = self.hubert(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
        logits        = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


HUBERT_TO_IDX = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5}


# ── Subset echilibrat ─────────────────────────────────────────────────────────
def build_balanced_subset(n_per_class=120, seed=42):
    """
    Construieste un subset echilibrat cu n_per_class exemple per clasa.
    Combina test.csv + val.csv, esantionand stratificat.
    """
    test_df = pd.read_csv(PROJECT_DIR / 'test.csv')
    val_df  = pd.read_csv(PROJECT_DIR / 'val.csv')
    all_df  = pd.concat([test_df, val_df], ignore_index=True)
    all_df  = all_df[all_df['emotion'].isin(EMOTION_CLASSES)].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    rows = []
    for emotion in EMOTION_CLASSES:
        subset = all_df[all_df['emotion'] == emotion]
        if len(subset) >= n_per_class:
            sampled = subset.sample(n=n_per_class, random_state=seed)
        else:
            # Oversampling cu replacement daca nu sunt destule
            sampled = subset.sample(n=n_per_class, replace=True, random_state=seed)
        rows.append(sampled)

    balanced = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed)
    total = len(balanced)
    pct   = total / 7441 * 100
    print(f"[INFO] Subset echilibrat: {n_per_class} exemple × {NUM_CLASSES} clase = {total} total ({pct:.1f}%)")
    for em in EMOTION_CLASSES:
        print(f"  {em}: {(balanced['emotion'] == em).sum()}")
    return balanced


# ── Inferenta video ───────────────────────────────────────────────────────────
def load_video_model(device):
    ckpt_path = MODELS_DIR / f'video_vit_haar_best_{NUM_CLASSES}cls.pt'
    if not ckpt_path.exists():
        ckpt_path = MODELS_DIR / 'video_vit_best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = VideoViViT(num_classes=NUM_CLASSES, embed_dim=VIDEO_EMBED_DIM,
                       dropout=VIDEO_DROPOUT).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[Video] {ckpt_path.name} | epoch={ckpt['epoch']} | val_acc={ckpt['val_acc']:.4f}")
    return model

def predict_video_row(video_path, model, device):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) == 0:
        return np.ones(NUM_CLASSES) / NUM_CLASSES

    # Haar crop
    mid  = frames[len(frames) // 2]
    h, w = mid.shape[:2]
    half = VIDEO_RESIZE // 2
    gray  = cv2.cvtColor(mid, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4, minSize=(30,30))
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        cx, cy = x+fw//2, y+fh//2
    else:
        cx, cy = w//2, h//2

    x1, y1, x2, y2 = cx-half, cy-half, cx+half, cy+half
    processed = []
    for frame in [frames[i] for i in np.linspace(0, len(frames)-1, VIDEO_NUM_FRAMES, dtype=int)]:
        pt = max(0, -y1); pb = max(0, y2-h); pl = max(0, -x1); pr = max(0, x2-w)
        crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt,pb),(pl,pr),(0,0)), mode='constant')
        if crop.shape[0] != VIDEO_RESIZE or crop.shape[1] != VIDEO_RESIZE:
            crop = cv2.resize(crop, (VIDEO_RESIZE, VIDEO_RESIZE))
        crop = crop.astype(np.float32)/255.0
        crop = (crop - MEAN) / STD
        processed.append(crop)

    arr    = np.stack(processed)
    tensor = torch.tensor(arr).permute(3,0,1,2).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    return F.softmax(logits, dim=-1).cpu().numpy()[0]


# ── Inferenta audio ───────────────────────────────────────────────────────────
def load_hubert_model(device):
    fe    = AutoFeatureExtractor.from_pretrained(str(HUBERT_MODEL_DIR))
    model = HubertForSpeechClassification.from_pretrained(str(HUBERT_MODEL_DIR)).to(device)
    model.eval()
    labels = [model.config.id2label[i] for i in range(NUM_CLASSES)]
    print(f"[Audio] HuBERT | labels={labels}")
    return fe, model, labels

def predict_audio_row(audio_path, fe, model, h_labels, device):
    try:
        audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    except:
        return np.ones(NUM_CLASSES) / NUM_CLASSES

    inputs = fe(audio, sampling_rate=16000, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    hp = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # Realiniaza
    pp = np.zeros(NUM_CLASSES, dtype=np.float32)
    MAP = {'angry':0,'disgust':1,'fear':2,'happy':3,'neutral':4,'sad':5}
    for i, hl in enumerate(h_labels):
        j = MAP.get(hl)
        if j is not None:
            pp[j] = hp[i]
    s = pp.sum()
    if s > 0: pp /= s
    return pp


# ── Matrice de confuzie estetica ──────────────────────────────────────────────
def plot_cm(cm, title, filename, n_per_class):
    """Ploteza o matrice de confuzie cu procente si conturi."""
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Colormap custom: alb → albastru inchis
    cmap = LinearSegmentedColormap.from_list(
        'emotion_cm',
        ['#FFFFFF', '#C8E6FA', '#5BAAD9', '#1565C0', '#0D3B6E'],
        N=256
    )

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    sns.heatmap(
        cm_pct,
        annot=False,
        fmt='.1f',
        cmap=cmap,
        vmin=0, vmax=100,
        linewidths=0.8,
        linecolor='#DDDDDD',
        square=True,
        ax=ax,
        cbar_kws={'shrink': 0.75, 'label': 'Procentaj (%)', 'format': '%.0f%%'}
    )

    # Adauga text in celule: procent + (count)
    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            pct = cm_pct[i, j]
            cnt = cm[i, j]
            txt_color = 'white' if pct > 55 else '#1A1A2E'
            weight = 'bold' if i == j else 'normal'
            ax.text(j + 0.5, i + 0.38,
                    f'{pct:.1f}%',
                    ha='center', va='center',
                    fontsize=11, color=txt_color,
                    fontweight=weight)
            ax.text(j + 0.5, i + 0.68,
                    f'({cnt})',
                    ha='center', va='center',
                    fontsize=8.5, color=txt_color,
                    alpha=0.85)

    ax.set_xticklabels(CLASS_LABELS, fontsize=11, rotation=30, ha='right', fontweight='medium')
    ax.set_yticklabels(CLASS_LABELS, fontsize=11, rotation=0, fontweight='medium')
    ax.set_xlabel('Clasă Prezisă', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('Clasă Reală', fontsize=13, fontweight='bold', labelpad=12)

    # Titlu cu acuratete
    acc = np.diag(cm).sum() / cm.sum() * 100
    ax.set_title(f'{title}\nAcuratețe: {acc:.2f}%  |  {n_per_class} exemple/clasă × {NUM_CLASSES} clase = {cm.sum()} total',
                 fontsize=13, fontweight='bold', pad=16, color='#1A1A2E')

    # Highlight diagonala
    for i in range(len(CLASS_LABELS)):
        ax.add_patch(mpatches.Rectangle((i, i), 1, 1,
                     fill=False, edgecolor='#FF6B35', lw=2.5))

    plt.tight_layout(pad=1.5)
    plt.savefig(filename, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Salvat: {filename}  (acc={acc:.2f}%)")
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    N_PER_CLASS = 120  # 120 × 6 = 720 exemple = ~9.7% din 7441
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Subset echilibrat
    df = build_balanced_subset(n_per_class=N_PER_CLASS)

    # Incarcare modele
    print("\n[INFO] Incarcare modele...")
    video_model = load_video_model(device)
    fe, hubert_model, h_labels = load_hubert_model(device)

    # Inferenta
    from tqdm import tqdm
    y_true, y_video, y_audio, y_fusion = [], [], [], []

    print(f"\n[INFO] Inferenta pe {len(df)} clipuri...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        true_idx = EMOTION_CLASSES.index(row['emotion'])
        y_true.append(true_idx)

        vp = predict_video_row(row['video_path'], video_model, device)
        ap = predict_audio_row(row['audio_path'], fe, hubert_model, h_labels, device)
        fp = VIDEO_WEIGHT * vp + AUDIO_WEIGHT * ap

        y_video.append(np.argmax(vp))
        y_audio.append(np.argmax(ap))
        y_fusion.append(np.argmax(fp))

    y_true   = np.array(y_true)
    y_video  = np.array(y_video)
    y_audio  = np.array(y_audio)
    y_fusion = np.array(y_fusion)

    # Matrici de confuzie
    cm_video  = confusion_matrix(y_true, y_video,  labels=list(range(NUM_CLASSES)))
    cm_audio  = confusion_matrix(y_true, y_audio,  labels=list(range(NUM_CLASSES)))
    cm_fusion = confusion_matrix(y_true, y_fusion, labels=list(range(NUM_CLASSES)))

    print(f"\n[INFO] Generare grafice...")
    acc_v = plot_cm(cm_video,  'Model Video — ViViT + Haar Cascade',
                    RESULTS_DIR / 'cm_video_haar_balanced.png', N_PER_CLASS)
    acc_a = plot_cm(cm_audio,  'Model Audio — HuBERT fine-tuned CREMA-D',
                    RESULTS_DIR / 'cm_audio_hubert_balanced.png', N_PER_CLASS)
    acc_f = plot_cm(cm_fusion, 'Fuziune Tardivă — ViViT (35%) + HuBERT (65%)',
                    RESULTS_DIR / 'cm_fusion_balanced.png', N_PER_CLASS)

    print(f"\n{'='*55}")
    print(f"  REZUMAT (subset echilibrat {N_PER_CLASS}/clasă)")
    print(f"  Video:   {acc_v:.2f}%")
    print(f"  Audio:   {acc_a:.2f}%")
    print(f"  Fuziune: {acc_f:.2f}%")
    print(f"  Imagini salvate in: {RESULTS_DIR}")
    print(f"{'='*55}")

if __name__ == '__main__':
    main()
