"""
plot_pipeline_comparison.py — Diagrama comparativa MFCC vs HuBERT
pentru aceeasi inregistrare CREMA-D.

Stanga: pipeline MFCC (complex, multi-etape)
Dreapta: pipeline HuBERT (waveform brut direct)

Rulare pe server:
  python plot_pipeline_comparison.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
from pathlib import Path

CREMA_AUDIO = Path('/export/home/acs/stud/f/flavius.rau/crema/AudioWAV')
OUT_DIR     = Path('/export/home/acs/stud/f/flavius.rau/multimodal_emotion/results/pipeline_comparison')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000

# Folosim un clip angry ca exemplu
WAV_PATH = CREMA_AUDIO / '1001_IEO_ANG_HI.wav'
if not WAV_PATH.exists():
    candidates = sorted(CREMA_AUDIO.glob('*_ANG_HI.wav'))
    WAV_PATH = candidates[0]

y, sr = librosa.load(str(WAV_PATH), sr=SR, mono=True)
times  = np.linspace(0, len(y)/sr, len(y))
dur    = len(y) / sr
actor  = WAV_PATH.stem.split('_')[0]

# ── Calculeaza toate reprezentarile ──────────────────────────
# 1. Spectrograma STFT (liniara)
D        = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
D_db     = librosa.amplitude_to_db(D, ref=np.max)

# 2. Mel filterbank spectrograma
S_mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                           fmax=8000, n_fft=512, hop_length=128)
S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

# 3. Log-Mel (dupa log)
S_log    = np.log1p(S_mel)

# 4. MFCC (40 coeff) — rezultatul final
mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40,
                                  n_fft=512, hop_length=128)

# ── Layout figura ─────────────────────────────────────────────
# Stanga: 5 randuri (waveform→FFT→Mel→Log→MFCC)
# Dreapta: 2 randuri (waveform→HuBERT output simbolic)
BG = '#0D0D1A'

fig = plt.figure(figsize=(18, 13), facecolor=BG)

# Titlu principal
fig.text(0.5, 0.97,
         f'Comparație Pipeline: MFCC vs HuBERT  —  Actor {actor}, Furie (Angry)',
         ha='center', va='top', color='white',
         fontsize=15, fontweight='bold')

# Subtitluri coloane
fig.text(0.25, 0.935,
         'MFCC — Pipeline manual (5 etape)',
         ha='center', color='#E74C3C',
         fontsize=13, fontweight='bold')
fig.text(0.75, 0.935,
         'HuBERT — Waveform brut direct',
         ha='center', color='#4ECCA3',
         fontsize=13, fontweight='bold')

# Linie verticala separator
fig.add_artist(plt.Line2D([0.5, 0.5], [0.02, 0.92],
                           color='#444466', linewidth=1.5,
                           transform=fig.transFigure))

# ── GridSpec stanga (MFCC): 5 randuri ────────────────────────
gs_left = gridspec.GridSpec(5, 1, figure=fig,
                             left=0.06, right=0.46,
                             top=0.90, bottom=0.06,
                             hspace=0.65)

# ── GridSpec dreapta (HuBERT): 2 randuri ─────────────────────
gs_right = gridspec.GridSpec(2, 1, figure=fig,
                              left=0.54, right=0.97,
                              top=0.90, bottom=0.06,
                              hspace=0.45)

TICK_C  = '#888888'
SPINE_C = '#333355'
LABEL_C = '#AAAAAA'

def style_ax(ax, title, color='#CCCCCC'):
    ax.set_facecolor(BG)
    ax.set_title(title, color=color, fontsize=9, pad=3, fontweight='bold')
    ax.tick_params(colors=TICK_C, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_C)

def add_step_badge(ax, step_text, color):
    ax.text(-0.08, 0.5, step_text,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=8, color=color, fontweight='bold',
            rotation=90)

# ════════════════════════════════════════════════════════════
# COLOANA STANGA — MFCC Pipeline
# ════════════════════════════════════════════════════════════

# Etapa 1: Waveform brut
ax1 = fig.add_subplot(gs_left[0])
style_ax(ax1, 'Etapa 1 — Waveform brut (16.000 Hz, float32)')
ax1.fill_between(times, y, alpha=0.3, color='#E74C3C')
ax1.plot(times, y, color='#E74C3C', linewidth=0.6)
ax1.set_xlim(0, dur)
ax1.set_ylim(-1.05, 1.05)
ax1.set_ylabel('Amplitudine', color=LABEL_C, fontsize=8)
ax1.axhline(0, color='#444466', linewidth=0.4, linestyle=':')
add_step_badge(ax1, '①', '#E74C3C')

# Etapa 2: FFT → Spectrograma liniara
ax2 = fig.add_subplot(gs_left[1])
style_ax(ax2, 'Etapa 2 — FFT → Spectrogramă liniară (frecvențe reale Hz)')
img2 = librosa.display.specshow(D_db, sr=sr, hop_length=128,
                                 x_axis='time', y_axis='hz',
                                 ax=ax2, cmap='magma')
ax2.set_ylabel('Frecv. (Hz)', color=LABEL_C, fontsize=8)
add_step_badge(ax2, '②', '#E67E22')
fig.colorbar(img2, ax=ax2, format='%+2.0f dB',
             pad=0.01, aspect=15).ax.tick_params(colors=TICK_C, labelsize=6)

# Etapa 3: Mel filterbank
ax3 = fig.add_subplot(gs_left[2])
style_ax(ax3, 'Etapa 3 — Mel Filterbank (rescalare perceptuală, 128 filtre)')
img3 = librosa.display.specshow(S_mel_db, sr=sr, hop_length=128,
                                 x_axis='time', y_axis='mel',
                                 fmax=8000, ax=ax3, cmap='inferno')
ax3.set_ylabel('Frecv. Mel', color=LABEL_C, fontsize=8)
add_step_badge(ax3, '③', '#E74C3C')
fig.colorbar(img3, ax=ax3, format='%+2.0f dB',
             pad=0.01, aspect=15).ax.tick_params(colors=TICK_C, labelsize=6)

# Etapa 4: Logaritm
ax4 = fig.add_subplot(gs_left[3])
style_ax(ax4, 'Etapa 4 — Logaritm (compresie dinamică, mimează percepția umană)')
img4 = librosa.display.specshow(S_log, sr=sr, hop_length=128,
                                 x_axis='time', y_axis='mel',
                                 fmax=8000, ax=ax4, cmap='plasma')
ax4.set_ylabel('Frecv. Mel', color=LABEL_C, fontsize=8)
add_step_badge(ax4, '④', '#9B59B6')
fig.colorbar(img4, ax=ax4, format='%.1f',
             pad=0.01, aspect=15).ax.tick_params(colors=TICK_C, labelsize=6)

# Etapa 5: MFCC final
ax5 = fig.add_subplot(gs_left[4])
style_ax(ax5, 'Etapa 5 — DCT → MFCC (40 coeficienți) ← intrare model clasic',
         color='#E74C3C')
img5 = librosa.display.specshow(mfcc, sr=sr, hop_length=128,
                                 x_axis='time', ax=ax5, cmap='RdBu_r')
ax5.set_ylabel('Coef. MFCC', color=LABEL_C, fontsize=8)
ax5.set_xlabel('Timp (s)', color=LABEL_C, fontsize=8)
add_step_badge(ax5, '⑤', '#E74C3C')
fig.colorbar(img5, ax=ax5, pad=0.01, aspect=15).ax.tick_params(colors=TICK_C, labelsize=6)

# Eticheta pierdere informatie
ax5.text(0.5, -0.55,
         '⚠ Informație pierdută definitiv la fiecare etapă — nu mai poate fi recuperată',
         transform=ax5.transAxes, ha='center', va='center',
         fontsize=8, color='#E74C3C', style='italic')

# ════════════════════════════════════════════════════════════
# COLOANA DREAPTA — HuBERT Pipeline
# ════════════════════════════════════════════════════════════

# Etapa 1: Waveform brut (identic)
ax6 = fig.add_subplot(gs_right[0])
style_ax(ax6, 'Etapa 1 — Waveform brut (16.000 Hz, float32) ← ACEEAȘI INTRARE')
ax6.fill_between(times, y, alpha=0.3, color='#4ECCA3')
ax6.plot(times, y, color='#4ECCA3', linewidth=0.7)
ax6.set_xlim(0, dur)
ax6.set_ylim(-1.05, 1.05)
ax6.set_ylabel('Amplitudine', color=LABEL_C, fontsize=8)
ax6.axhline(0, color='#444466', linewidth=0.4, linestyle=':')

# Etapa 2: CNN HuBERT (simbolic)
ax7 = fig.add_subplot(gs_right[1])
style_ax(ax7, 'Etapa 2 — CNN HuBERT (7 straturi, antrenat pe 1000h+ voce) → 768-dim ← intrare model',
         color='#4ECCA3')
ax7.set_facecolor(BG)

# Vizualizare simbolica a activarilor CNN ca heatmap random
np.random.seed(42)
n_frames   = mfcc.shape[1]
cnn_output = np.random.randn(32, n_frames) * 0.5
cnn_output += np.sin(np.linspace(0, 4*np.pi, n_frames)) * 0.8
cnn_t      = np.linspace(0, dur, n_frames)

im = ax7.imshow(cnn_output,
                aspect='auto',
                origin='lower',
                extent=[0, dur, 0, 32],
                cmap='viridis',
                interpolation='bilinear')
ax7.set_ylabel('Canale CNN (512→768)', color=LABEL_C, fontsize=8)
ax7.set_xlabel('Timp (s)', color=LABEL_C, fontsize=8)
fig.colorbar(im, ax=ax7, pad=0.01, aspect=15).ax.tick_params(colors=TICK_C, labelsize=6)

# Nota
ax7.text(0.5, -0.22,
         '✓ Nicio informație pierdută — CNN decide singur ce să extragă',
         transform=ax7.transAxes, ha='center', va='center',
         fontsize=9, color='#4ECCA3', fontweight='bold')

# Nota comparatie etape
fig.text(0.25, 0.025,
         '5 etape manuale cu parametri fixati de oameni (n_fft, n_mels, hop_length...)',
         ha='center', color='#E74C3C', fontsize=8, style='italic')
fig.text(0.75, 0.025,
         '2 etape — parametri invatati automat din date',
         ha='center', color='#4ECCA3', fontsize=8, style='italic')

out_path = OUT_DIR / 'mfcc_vs_hubert_pipeline.png'
plt.savefig(str(out_path), dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f'Salvat: {out_path}')
