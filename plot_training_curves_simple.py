"""
plot_training_curves_simple.py — Grafice simple tip articol stiintific
pentru ViViT Haar si HuBERT. Stil curat, alb, fara decoratii.

Rulare pe server:
  python plot_training_curves_simple.py
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path('/export/home/acs/stud/f/flavius.rau/multimodal_emotion/results/training_curves_simple')
OUT_DIR.mkdir(parents=True, exist_ok=True)

HUBERT_HISTORY = Path('/export/home/acs/stud/f/flavius.rau/multimodal_emotion/runs/chinese_hubert_samespeaker/history.json')

# ── Date ViViT Haar ───────────────────────────────────────────
vivit_haar = [
    ( 1,1.1240,0.3210,1.5117,0.3696),( 2,0.9416,0.4358,1.2402,0.5412),
    ( 3,0.8362,0.5152,1.1972,0.5480),( 4,0.7412,0.5942,1.0494,0.6500),
    ( 5,0.6679,0.6470,0.9918,0.6627),( 6,1.0687,0.6228,0.9112,0.6610),
    ( 7,1.0356,0.6331,0.8292,0.6856),( 8,0.9963,0.6479,0.8260,0.6975),
    ( 9,0.9570,0.6663,0.7528,0.7188),(10,0.8955,0.6907,0.9920,0.6491),
    (11,0.8567,0.7153,0.6808,0.7562),(12,0.7769,0.7506,0.8482,0.7137),
    (13,0.7598,0.7612,0.6449,0.7732),(14,0.6922,0.7928,0.6535,0.7825),
    (15,0.6359,0.8227,0.6389,0.7952),(16,0.6044,0.8308,0.5683,0.8165),
    (17,0.5740,0.8555,0.7385,0.7621),(18,0.5188,0.8666,0.6200,0.8054),
    (19,0.4764,0.8849,0.8085,0.8267),(20,0.4614,0.8965,0.8424,0.8097),
    (21,0.4309,0.9022,0.6143,0.8352),(22,0.4277,0.9064,0.7458,0.8267),
    (23,0.4045,0.9167,0.8115,0.8199),(24,0.3739,0.9266,0.9530,0.8029),
    (25,0.3696,0.9262,0.8535,0.8182),(26,0.3685,0.9240,0.7184,0.8343),
    (27,0.3682,0.9273,0.8689,0.8326),(28,0.3421,0.9352,0.8868,0.8199),
    (29,0.3403,0.9367,0.9433,0.8292),
]
v_ep    = [r[0] for r in vivit_haar]
v_tloss = [r[1] for r in vivit_haar]
v_vloss = [r[3] for r in vivit_haar]
v_vacc  = [r[4] for r in vivit_haar]

# ── Date HuBERT ───────────────────────────────────────────────
with open(HUBERT_HISTORY) as f:
    hh = json.load(f)
h_ep    = [r['epoch']        for r in hh]
h_tloss = [r['train_loss']   for r in hh]
h_vloss = [r['val_loss']     for r in hh]
h_vacc  = [r['val_accuracy'] for r in hh]

# ── Stil comun ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.linewidth':   0.8,
    'axes.edgecolor':   '#333333',
    'axes.facecolor':   '#F5F6FA',
    'figure.facecolor': 'white',
    'xtick.direction':  'out',
    'ytick.direction':  'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'lines.linewidth':  1.5,
    'lines.color':      '#4472C4',
})

def make_pair(epochs, loss_vals, acc_vals,
              loss_title, acc_title, fname, acc_lim=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    fig.subplots_adjust(wspace=0.38, left=0.10, right=0.97,
                        top=0.88, bottom=0.16)

    # ── Loss ──────────────────────────────────────────────────
    ax1.plot(epochs, loss_vals, color='#4472C4', linewidth=1.6)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title(loss_title, fontsize=10, pad=6)
    ax1.set_xlim(epochs[0], epochs[-1])
    ax1.set_ylim(bottom=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ticks la fiecare ~5 epoci
    step = 5
    ticks = [epochs[0]] + [e for e in epochs if e % step == 0]
    ax1.set_xticks(sorted(set(ticks)))

    # ── Accuracy ──────────────────────────────────────────────
    ax2.plot(epochs, acc_vals, color='#4472C4', linewidth=1.6)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title(acc_title, fontsize=10, pad=6)
    ax2.set_xlim(epochs[0], epochs[-1])
    if acc_lim:
        ax2.set_ylim(acc_lim)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xticks(sorted(set(ticks)))

    fig.savefig(str(OUT_DIR / fname), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Salvat: {fname}')


# ── Fig 1: ViViT Haar ─────────────────────────────────────────
make_pair(
    epochs    = v_ep,
    loss_vals = v_tloss,
    acc_vals  = v_vacc,
    loss_title= 'Fig. 7. Loss function values through the\ntraining process (ViViT + Haar)',
    acc_title = 'Fig. 8. Test accuracy evolution on the\nvalidation dataset (ViViT + Haar)',
    fname     = 'vivit_haar_curves_simple.png',
    acc_lim   = (0.30, 0.90),
)

# ── Fig 2: HuBERT ─────────────────────────────────────────────
make_pair(
    epochs    = h_ep,
    loss_vals = h_tloss,
    acc_vals  = h_vacc,
    loss_title= 'Fig. 9. Loss function values through the\ntraining process (HuBERT)',
    acc_title = 'Fig. 10. Test accuracy evolution on the\nvalidation dataset (HuBERT)',
    fname     = 'hubert_curves_simple.png',
    acc_lim   = (0.40, 0.85),
)

print(f'\nGata! 2 figuri in: {OUT_DIR}')
