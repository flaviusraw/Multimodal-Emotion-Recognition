"""
plot_hubert_cnn_real.py — Vizualizeaza activarile CNN reale din HuBERT
pentru un clip CREMA-D, alaturi de waveform-ul de intrare.

Rulare pe server:
  python plot_hubert_cnn_real.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import torch
import torch.nn as nn
from pathlib import Path
from transformers import HubertModel, HubertPreTrainedModel, AutoFeatureExtractor
import sys
sys.path.insert(0, '/export/home/acs/stud/f/flavius.rau/multimodal_emotion')

CREMA_AUDIO  = Path('/export/home/acs/stud/f/flavius.rau/crema/AudioWAV')
HUBERT_DIR   = Path('/export/home/acs/stud/f/flavius.rau/multimodal_emotion/runs/chinese_hubert_samespeaker/best_model')
OUT_DIR      = Path('/export/home/acs/stud/f/flavius.rau/multimodal_emotion/results/hubert_cnn_real')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000

CLIPS = {
    'angry':   '1001_IEO_ANG_HI.wav',
    'happy':   '1063_IEO_HAP_HI.wav',
    'sad':     '1050_IEO_SAD_HI.wav',
    'neutral': '1030_IEO_NEU_XX.wav',
    'fearful': '1045_IEO_FEA_HI.wav',
    'disgust': '1020_IEO_DIS_HI.wav',
}

EMOTION_COLOR = {
    'angry':   '#E74C3C',
    'happy':   '#F1C40F',
    'sad':     '#3498DB',
    'neutral': '#95A5A6',
    'fearful': '#9B59B6',
    'disgust': '#27AE60',
}

EMOTION_LABEL = {
    'angry':   'Furie (Angry)',
    'happy':   'Fericire (Happy)',
    'sad':     'Tristete (Sad)',
    'neutral': 'Neutru (Neutral)',
    'fearful': 'Frica (Fearful)',
    'disgust': 'Dezgust (Disgust)',
}

# ── Incarca modelul HuBERT o singura data ────────────────────
print('[INFO] Incarcare HuBERT...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Incarca doar feature_extractor CNN (primele 7 straturi conv)
hubert = HubertModel.from_pretrained(str(HUBERT_DIR)).to(device)
hubert.eval()
print(f'[INFO] Model incarcat pe {device}')

fe = AutoFeatureExtractor.from_pretrained(str(HUBERT_DIR))

BG = '#0D0D1A'

for emotion, wav_name in CLIPS.items():
    wav_path = CREMA_AUDIO / wav_name
    if not wav_path.exists():
        code = {'angry':'ANG','disgust':'DIS','fearful':'FEA',
                'happy':'HAP','neutral':'NEU','sad':'SAD'}[emotion]
        candidates = sorted(CREMA_AUDIO.glob(f'*_{code}_HI.wav'))
        if not candidates:
            candidates = sorted(CREMA_AUDIO.glob(f'*_{code}_*.wav'))
        if not candidates:
            print(f'SKIP {emotion}')
            continue
        wav_path = candidates[4]

    actor = wav_path.stem.split('_')[0]
    color = EMOTION_COLOR[emotion]

    # Incarca audio
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    times = np.linspace(0, len(y)/SR, len(y))
    dur   = len(y) / SR

    # ── Extrage activarile CNN reale ─────────────────────────
    inputs = fe(y, sampling_rate=SR, return_tensors='pt', padding=True)
    input_values = inputs['input_values'].to(device)

    with torch.no_grad():
        # feature_extractor = CNN cu 7 straturi
        # output: (1, 512, T_frames)
        cnn_out = hubert.feature_extractor(input_values)

    # (1, 512, T) → (512, T) → numpy
    cnn_np = cnn_out.squeeze(0).cpu().numpy()
    print(f'  {emotion}: CNN output shape = {cnn_np.shape}')

    n_channels, n_frames = cnn_np.shape
    frame_times = np.linspace(0, dur, n_frames)

    # Selecteaza primele 32 canale pentru vizualizare
    N_SHOW = 32
    cnn_show = cnn_np[:N_SHOW, :]

    # Normalizeaza per canal pentru vizualizare clara
    for i in range(N_SHOW):
        mn, mx = cnn_show[i].min(), cnn_show[i].max()
        if mx > mn:
            cnn_show[i] = (cnn_show[i] - mn) / (mx - mn)

    # Mel spectrograma pentru comparatie
    S_mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64,
                                               fmax=8000, n_fft=512, hop_length=128)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

    # ── Layout figura ─────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), facecolor=BG)

    fig.suptitle(
        f"HuBERT CNN — Activări Reale vs Waveform Brut\n"
        f"{EMOTION_LABEL[emotion]}  —  Actor {actor}",
        color='white', fontsize=13, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           top=0.91, bottom=0.07,
                           left=0.08, right=0.97)

    TICK_C  = '#888888'
    SPINE_C = '#333355'
    LAB_C   = '#AAAAAA'

    def style_ax(ax, title, title_color='#CCCCCC'):
        ax.set_facecolor(BG)
        ax.set_title(title, color=title_color, fontsize=9.5,
                     pad=4, fontweight='bold')
        ax.tick_params(colors=TICK_C, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_C)

    # ── [0,0] Waveform intrare ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, f'① Intrare HuBERT — Waveform brut (float32, {SR} Hz)', color)
    ax1.fill_between(times, y, alpha=0.3, color=color)
    ax1.plot(times, y, color=color, linewidth=0.7)
    ax1.set_xlim(0, dur)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel('Amplitudine', color=LAB_C, fontsize=8)
    ax1.set_xlabel('Timp (s)', color=LAB_C, fontsize=8)
    ax1.axhline(0, color='#444466', linewidth=0.4, linestyle=':')
    ax1.text(0.98, 0.95, f'{len(y):,} samples',
             transform=ax1.transAxes, ha='right', va='top',
             color='#666688', fontsize=8, fontfamily='monospace')

    # ── [0,1] Mel spectrograma (pentru comparatie vizuala) ───
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, '→ Mel Spectrogramă (nu intră în HuBERT — doar comparație vizuală)',
             '#888888')
    img2 = librosa.display.specshow(S_mel_db, sr=SR, hop_length=128,
                                     x_axis='time', y_axis='mel',
                                     fmax=8000, ax=ax2, cmap='magma')
    ax2.set_ylabel('Frecv. (Hz)', color=LAB_C, fontsize=8)
    ax2.set_xlabel('Timp (s)', color=LAB_C, fontsize=8)
    cb2 = fig.colorbar(img2, ax=ax2, format='%+2.0f dB', pad=0.01, aspect=20)
    cb2.ax.tick_params(colors=TICK_C, labelsize=7)
    # watermark
    ax2.text(0.5, 0.5, 'NU INTRĂ ÎN HUBERT', transform=ax2.transAxes,
             ha='center', va='center', fontsize=14, color='white',
             alpha=0.15, fontweight='bold', rotation=20)

    # ── [1,0:2] Activarile CNN reale (primele 32 canale) ─────
    ax3 = fig.add_subplot(gs[1, :])
    style_ax(ax3,
             f'② CNN Feature Extractor HuBERT — Activări reale '
             f'(primele {N_SHOW} din 512 canale, {n_frames} cadre × ~20ms)',
             '#4ECCA3')
    img3 = ax3.imshow(cnn_show,
                      aspect='auto',
                      origin='lower',
                      extent=[0, dur, 0, N_SHOW],
                      cmap='viridis',
                      interpolation='bilinear',
                      vmin=0, vmax=1)
    ax3.set_ylabel(f'Canal CNN (0-{N_SHOW-1})', color=LAB_C, fontsize=8)
    ax3.set_xlabel('Timp (s)', color=LAB_C, fontsize=8)
    cb3 = fig.colorbar(img3, ax=ax3, pad=0.01, aspect=30)
    cb3.ax.tick_params(colors=TICK_C, labelsize=7)
    cb3.set_label('Activare normalizată', color=LAB_C, fontsize=7)

    # ── [2,0] Distributia activarilor per canal ──────────────
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4, 'Distribuție activări medii per canal (primele 32)', '#4ECCA3')
    means = cnn_np[:N_SHOW].mean(axis=1)
    bars  = ax4.bar(range(N_SHOW), means,
                    color=[plt.cm.viridis(v/means.max()) for v in means],
                    width=0.8, edgecolor='none')
    ax4.set_xlabel('Canal CNN', color=LAB_C, fontsize=8)
    ax4.set_ylabel('Activare medie', color=LAB_C, fontsize=8)
    ax4.set_xlim(-0.5, N_SHOW - 0.5)

    # ── [2,1] Energie totala CNN in timp ─────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5, 'Energie totală CNN în timp — ce "vede" HuBERT per cadru', '#4ECCA3')
    energy = np.abs(cnn_np).mean(axis=0)
    ax5.fill_between(frame_times, energy, alpha=0.4, color='#4ECCA3')
    ax5.plot(frame_times, energy, color='#4ECCA3', linewidth=1)
    ax5.set_xlabel('Timp (s)', color=LAB_C, fontsize=8)
    ax5.set_ylabel('Energie medie', color=LAB_C, fontsize=8)
    ax5.set_xlim(0, dur)

    # Nota jos
    fig.text(0.5, 0.015,
             f'CNN output: ({n_channels} canale × {n_frames} cadre) → '
             f'proiectat la 768 dim → Transformer × 12 straturi → Mean Pooling → Head clasificare',
             ha='center', color='#666688', fontsize=8, fontfamily='monospace')

    out_path = OUT_DIR / f'{emotion}_cnn_real.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Salvat: {out_path.name}')

print(f'\nGata! {len(CLIPS)} imagini in: {OUT_DIR}')
