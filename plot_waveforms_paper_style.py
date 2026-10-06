"""
plot_waveforms_paper_style.py — 6 imagini waveform stil articol stiintific
Simplu, curat, albastru pe alb, ca in figura din paper.

Rulare pe server:
  python plot_waveforms_paper_style.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
from pathlib import Path

CREMA_AUDIO = Path('/export/home/acs/stud/f/flavius.rau/crema/AudioWAV')
OUT_DIR     = Path('/export/home/acs/stud/f/flavius.rau/multimodal_emotion/results/waveforms_paper')
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

LABELS = {
    'angry':   'Angry',
    'disgust': 'Disgust',
    'fearful': 'Fearful',
    'happy':   'Happy',
    'neutral': 'Neutral',
    'sad':     'Sad',
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

    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    times = np.linspace(0, len(y)/SR, len(y))

    fig, ax = plt.subplots(figsize=(5.0, 2.8), facecolor='white')
    ax.set_facecolor('white')

    # Fill + line
    ax.fill_between(times, y, -y, alpha=0.18, color='#2E75B6')
    ax.fill_between(times, y,      alpha=0.55, color='#2E75B6')
    ax.fill_between(times, -np.abs(y), alpha=0.55, color='#2E75B6')
    ax.plot(times, y, color='#1F5FA6', linewidth=0.6, alpha=0.9)

    ax.set_xlim(0, len(y)/SR)
    ax.set_ylim(-1.05, 1.05)

    # Axe vizibile
    ax.set_xlabel('Time (s)', fontsize=10, color='#333333')
    ax.set_ylabel('Amplitude', fontsize=10, color='#333333')
    ax.set_title(LABELS[emotion], fontsize=11, fontweight='bold',
                 color='#222222', pad=6)

    # Ticks
    ax.tick_params(axis='both', labelsize=8, colors='#444444',
                   direction='out', length=3, width=0.7)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    # Spines — doar stanga si jos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_linewidth(0.7)
    ax.spines['bottom'].set_color('#888888')

    # Linie zero
    ax.axhline(0, color='#AAAAAA', linewidth=0.5, linestyle='--')

    plt.tight_layout(pad=0.8)
    out_path = OUT_DIR / f'{emotion}_waveform.png'
    plt.savefig(str(out_path), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f'  {emotion:8s} -> {out_path.name}')

print(f'\nGata! 6 imagini in: {OUT_DIR}')
