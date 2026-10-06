"""
plot_waveform_hubert.py — Vizualizeaza ce primeste HuBERT ca intrare:
waveform brut float32 16kHz, pentru cele 6 emotii CREMA-D.

Rulare pe server:
  python plot_waveform_hubert.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
from pathlib import Path

CREMA_AUDIO = Path('/export/home/acs/stud/f/flavius.rau/crema/AudioWAV')
OUT_DIR     = Path('/export/home/acs/stud/f/flavius.rau/multimodal_emotion/results/waveform_hubert')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000

CLIPS = {
    'angry':   '1001_IEO_ANG_HI.wav',
    'disgust': '1020_IEO_DIS_HI.wav',
    'fearful': '1045_IEO_FEA_HI.wav',
    'happy':   '1063_IEO_HAP_HI.wav',
    'neutral': '1030_IEO_NEU_XX.wav',
    'sad':     '1050_IEO_SAD_HI.wav',
}

EMOTION_STYLE = {
    'angry':   {'color': '#E74C3C', 'label': 'Furie (Angry)'},
    'disgust': {'color': '#27AE60', 'label': 'Dezgust (Disgust)'},
    'fearful': {'color': '#9B59B6', 'label': 'Frica (Fearful)'},
    'happy':   {'color': '#F1C40F', 'label': 'Fericire (Happy)'},
    'neutral': {'color': '#95A5A6', 'label': 'Neutru (Neutral)'},
    'sad':     {'color': '#3498DB', 'label': 'Tristete (Sad)'},
}

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
        wav_name = wav_path.name

    style  = EMOTION_STYLE[emotion]
    color  = style['color']
    y, sr  = librosa.load(str(wav_path), sr=SR, mono=True)
    times  = np.linspace(0, len(y)/sr, len(y))
    dur    = len(y) / sr
    actor  = wav_name.split('_')[0]

    fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0D0D1A')
    ax.set_facecolor('#0D0D1A')

    ax.fill_between(times, y, alpha=0.3, color=color)
    ax.plot(times, y, color=color, linewidth=0.7, alpha=0.95)

    ax.set_xlim(0, dur)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel('Timp (s)', color='#AAAAAA', fontsize=10)
    ax.set_ylabel('Amplitudine', color='#AAAAAA', fontsize=10)
    ax.set_title(f"{style['label']}  —  Actor {actor}  |  "
                 f"Intrare HuBERT: waveform brut float32 @ {SR} Hz",
                 color='white', fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#888888', labelsize=9)
    ax.axhline(0, color='#444466', linewidth=0.5, linestyle=':')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    # Statistici
    rms = float(np.sqrt(np.mean(y**2)))
    ax.text(0.99, 0.95,
            f'dur={dur:.2f}s  |  samples={len(y)}  |  RMS={rms:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            color='#666688', fontsize=8, fontfamily='monospace')

    plt.tight_layout(pad=0.8)
    out_path = OUT_DIR / f'{emotion}_waveform.png'
    plt.savefig(str(out_path), dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  {emotion:8s} -> {out_path.name}')

print(f'\nGata! 6 imagini in: {OUT_DIR}')
