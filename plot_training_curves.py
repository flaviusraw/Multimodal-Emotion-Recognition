"""
plot_training_curves.py — Grafice loss si accuracy pentru ViViT Haar si HuBERT
Date 100% reale din fisierele de output de antrenare.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path('/export/home/acs/stud/f/flavius.rau/multimodal_emotion/results/training_curves')
OUT_DIR.mkdir(parents=True, exist_ok=True)
HUBERT_HISTORY = Path('/export/home/acs/stud/f/flavius.rau/multimodal_emotion/runs/chinese_hubert_samespeaker/history.json')

BG = '#0D0D1A'; TICK_C = '#888888'; GRID_C = '#222244'; LAB_C = '#AAAAAA'

# ── ViViT Haar — date reale din haaroutputvideo.TXT ──────────
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
v_ep   = [r[0] for r in vivit_haar]
v_tloss= [r[1] for r in vivit_haar]; v_tacc = [r[2] for r in vivit_haar]
v_vloss= [r[3] for r in vivit_haar]; v_vacc = [r[4] for r in vivit_haar]
BEST_V_EP=21; BEST_V_VAL=0.8352; BEST_V_TEST=0.8309

# ── HuBERT — din history.json + outputaudio.txt ──────────────
with open(HUBERT_HISTORY) as f:
    hh = json.load(f)
h_ep   = [r['epoch']        for r in hh]
h_tloss= [r['train_loss']   for r in hh]
h_vloss= [r['val_loss']     for r in hh]
h_vacc = [r['val_accuracy'] for r in hh]
h_vf1  = [r['val_f1_macro'] for r in hh]
BEST_H_EP=11; BEST_H_VAL=0.8068; BEST_H_TEST=0.8566

def style_ax(ax, title, tc='#CCCCCC'):
    ax.set_facecolor(BG)
    ax.set_title(title, color=tc, fontsize=10, pad=6, fontweight='bold')
    ax.tick_params(colors=TICK_C, labelsize=9)
    ax.xaxis.label.set_color(LAB_C); ax.yaxis.label.set_color(LAB_C)
    for sp in ax.spines.values(): sp.set_edgecolor('#333355')
    ax.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.7)

# ════ FIG 1: ViViT Haar ════════════════════════════════════
fig1,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5),facecolor=BG)
fig1.suptitle('ViViT + Haar Cascade — Curbe de Antrenare  (29 epoci, early stopping)',
              color='white',fontsize=13,fontweight='bold')
fig1.patch.set_facecolor(BG)
plt.subplots_adjust(wspace=0.30,left=0.08,right=0.97,top=0.88,bottom=0.12)

style_ax(ax1,'Funcția de pierdere (Loss) — Train vs Validare')
ax1.plot(v_ep,v_tloss,color='#E74C3C',lw=2,marker='o',ms=3,label='Train loss')
ax1.plot(v_ep,v_vloss,color='#3498DB',lw=2,marker='s',ms=3,label='Val loss',ls='--')
ax1.axvline(x=5.5,color='#888899',lw=1.2,ls=':',alpha=0.7)
ax1.text(3,1.48,'Warm-up',color='#888899',fontsize=8,ha='center')
ax1.text(17,1.48,'Fine-tuning',color='#888899',fontsize=8,ha='center')
ax1.axvline(x=BEST_V_EP,color='#F1C40F',lw=1.4,ls='--',alpha=0.8)
ax1.text(BEST_V_EP+0.3,0.62,f'Best ep {BEST_V_EP}',color='#F1C40F',fontsize=8)
ax1.set_xlabel('Epocă'); ax1.set_ylabel('Loss'); ax1.set_xlim(0.5,29.5)
ax1.legend(facecolor='#1A1A2E',edgecolor='none',labelcolor='white',fontsize=9)

style_ax(ax2,'Acuratețe pe Validare și Train')
ax2.plot(v_ep,[a*100 for a in v_vacc],color='#4ECCA3',lw=2,marker='o',ms=3,label='Val accuracy')
ax2.plot(v_ep,[a*100 for a in v_tacc],color='#E74C3C',lw=1.5,marker='s',ms=3,
         label='Train accuracy',ls='--',alpha=0.7)
ax2.axvline(x=5.5,color='#888899',lw=1.2,ls=':',alpha=0.7)
ax2.axvline(x=BEST_V_EP,color='#F1C40F',lw=1.4,ls='--',alpha=0.8)
ax2.axhline(y=BEST_V_VAL*100,color='#F1C40F',lw=0.8,ls=':',alpha=0.5)
ax2.text(BEST_V_EP+0.3,BEST_V_VAL*100+0.8,
         f'Val: {BEST_V_VAL*100:.2f}%\nTest: {BEST_V_TEST*100:.2f}%',
         color='#F1C40F',fontsize=8)
ax2.set_xlabel('Epocă'); ax2.set_ylabel('Acuratețe (%)'); ax2.set_xlim(0.5,29.5); ax2.set_ylim(25,105)
ax2.legend(facecolor='#1A1A2E',edgecolor='none',labelcolor='white',fontsize=9)

fig1.savefig(str(OUT_DIR/'vivit_haar_training_curves.png'),dpi=160,bbox_inches='tight',facecolor=BG)
plt.close(fig1); print('  Salvat: vivit_haar_training_curves.png')

# ════ FIG 2: HuBERT ════════════════════════════════════════
fig2,(ax3,ax4)=plt.subplots(1,2,figsize=(13,5),facecolor=BG)
fig2.suptitle('HuBERT Fine-tuned CREMA-D — Curbe de Antrenare  (16 epoci, early stopping)',
              color='white',fontsize=13,fontweight='bold')
fig2.patch.set_facecolor(BG)
plt.subplots_adjust(wspace=0.30,left=0.08,right=0.97,top=0.88,bottom=0.12)

style_ax(ax3,'Funcția de pierdere (Loss) — Train vs Validare')
ax3.plot(h_ep,h_tloss,color='#E74C3C',lw=2,marker='o',ms=3,label='Train loss')
ax3.plot(h_ep,h_vloss,color='#3498DB',lw=2,marker='s',ms=3,label='Val loss',ls='--')
ax3.axvline(x=BEST_H_EP,color='#F1C40F',lw=1.4,ls='--',alpha=0.8)
ax3.text(BEST_H_EP+0.2,max(h_vloss)*0.96,f'Best ep {BEST_H_EP}',color='#F1C40F',fontsize=8)
ax3.axvline(x=16.5,color='#888899',lw=1.2,ls=':',alpha=0.7)
ax3.text(15.8,max(h_vloss)*0.84,'Early\nstopping',color='#888899',fontsize=7,ha='right')
ax3.set_xlabel('Epocă'); ax3.set_ylabel('Loss'); ax3.set_xlim(0.5,16.5)
ax3.legend(facecolor='#1A1A2E',edgecolor='none',labelcolor='white',fontsize=9)

style_ax(ax4,'Acuratețe și F1 Macro pe Validare')
ax4.plot(h_ep,[a*100 for a in h_vacc],color='#4ECCA3',lw=2,marker='o',ms=3,label='Val accuracy')
ax4.plot(h_ep,[f*100 for f in h_vf1], color='#9B59B6',lw=2,marker='s',ms=3,label='Val F1 macro',ls='--')
ax4.axvline(x=BEST_H_EP,color='#F1C40F',lw=1.4,ls='--',alpha=0.8)
ax4.axhline(y=BEST_H_VAL*100,color='#F1C40F',lw=0.8,ls=':',alpha=0.5)
ax4.text(BEST_H_EP+0.2,BEST_H_VAL*100+0.8,
         f'Val: {BEST_H_VAL*100:.2f}%\nTest: {BEST_H_TEST*100:.2f}%',
         color='#F1C40F',fontsize=8)
ax4.axvline(x=16.5,color='#888899',lw=1.2,ls=':',alpha=0.7)
ax4.set_xlabel('Epocă'); ax4.set_ylabel('(%)'); ax4.set_xlim(0.5,16.5); ax4.set_ylim(35,105)
ax4.legend(facecolor='#1A1A2E',edgecolor='none',labelcolor='white',fontsize=9)

fig2.savefig(str(OUT_DIR/'hubert_training_curves.png'),dpi=160,bbox_inches='tight',facecolor=BG)
plt.close(fig2); print('  Salvat: hubert_training_curves.png')

# ════ FIG 3: Bar chart comparatie finala ════════════════════
fig3,ax5=plt.subplots(1,1,figsize=(10,5),facecolor=BG)
fig3.suptitle('Performanță Finală pe Test Set — Toate Configurațiile',
              color='white',fontsize=13,fontweight='bold')
fig3.patch.set_facecolor(BG)
plt.subplots_adjust(left=0.09,right=0.97,top=0.88,bottom=0.18)
style_ax(ax5,'')

models=['ViViT\nCenter Crop\n(ep 27)','ViViT\nHaar Cascade\n(ep 21)',
        'HuBERT\n(ep 11)','Fuziune\n35% Video\n65% Audio']
accs=[78.43,83.09,85.66,90.20]; f1s=[76.49,81.59,84.07,88.86]
colors=['#C0392B','#E67E22','#1ABC9C','#F1C40F']
x=np.arange(len(models)); w=0.38

b1=ax5.bar(x-w/2,accs,w,color=colors,alpha=0.9,edgecolor='none',label='Accuracy (%)',zorder=3)
b2=ax5.bar(x+w/2,f1s, w,color=colors,alpha=0.55,edgecolor='none',label='F1 Macro (%)',zorder=3,hatch='///')

for bar,val in zip(b1,accs):
    ax5.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
             f'{val:.2f}%',ha='center',va='bottom',color='white',fontsize=9,fontweight='bold')
for bar,val in zip(b2,f1s):
    ax5.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
             f'{val:.2f}%',ha='center',va='bottom',color='white',fontsize=8.5)

ax5.set_xticks(x); ax5.set_xticklabels(models,color='#CCCCCC',fontsize=9)
ax5.set_ylabel('(%)',color=LAB_C); ax5.set_ylim(60,100)
ax5.legend(facecolor='#1A1A2E',edgecolor='none',labelcolor='white',fontsize=9,loc='lower right')

fig3.savefig(str(OUT_DIR/'final_performance_comparison.png'),dpi=160,bbox_inches='tight',facecolor=BG)
plt.close(fig3); print('  Salvat: final_performance_comparison.png')
print(f'\nGata! 3 figuri in: {OUT_DIR}')
