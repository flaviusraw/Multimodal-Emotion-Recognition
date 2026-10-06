"""
test_hubert_120.py — Evalueaza modelul HuBERT pe 120 clipuri din test.csv
pentru a verifica acuratetea de 85.66%.

Rulare pe server:
  python test_hubert_120.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoFeatureExtractor, HubertPreTrainedModel, HubertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from tqdm import tqdm
import sys
sys.path.insert(0, '/export/home/acs/stud/f/flavius.rau/multimodal_emotion')
from config import EMOTION_CLASSES, NUM_CLASSES, PROJECT_DIR

HUBERT_DIR = Path('/export/home/acs/stud/f/flavius.rau/multimodal_emotion/runs/chinese_hubert_samespeaker/best_model')
SR = 16000
N_PER_CLASS = 120  # 120 × 6 = 720 clipuri

HUBERT_TO_IDX = {
    'angry': 0, 'disgust': 1, 'fear': 2,
    'happy': 3, 'neutral': 4, 'sad':  5
}


# ── HuBERT arhitectura custom ─────────────────────────────────
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
        self.hubert              = HubertModel(config)
        self.classifier          = HubertClassificationHead(config)
        self._tied_weights_keys  = []
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


# ── Subset echilibrat din test.csv ────────────────────────────
def build_subset(csv_path, n_per_class=120, seed=42):
    df = pd.read_csv(csv_path)
    df = df[df['emotion'].isin(EMOTION_CLASSES)].reset_index(drop=True)

    rows = []
    for emotion in EMOTION_CLASSES:
        sub = df[df['emotion'] == emotion]
        if len(sub) >= n_per_class:
            sampled = sub.sample(n=n_per_class, random_state=seed)
        else:
            # oversampling daca nu sunt destule
            sampled = sub.sample(n=n_per_class, replace=True, random_state=seed)
            print(f'  [WARN] {emotion}: doar {len(sub)} clipuri, oversampleed la {n_per_class}')
        rows.append(sampled)

    balanced = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed)
    print(f'  Subset: {len(balanced)} clipuri ({n_per_class}/clasa × {NUM_CLASSES} clase)')
    for em in EMOTION_CLASSES:
        print(f'    {em}: {(balanced["emotion"] == em).sum()}')
    return balanced


# ── Predictie audio ───────────────────────────────────────────
@torch.no_grad()
def predict_audio(audio_path, feature_extractor, model, hubert_labels, device):
    try:
        audio, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    except Exception as e:
        print(f'  [WARN] Nu pot citi {audio_path}: {e}')
        return None

    inputs = feature_extractor(
        audio, sampling_rate=SR, return_tensors='pt', padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    hubert_probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # Realiniaza la EMOTION_CLASSES
    project_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, hlabel in enumerate(hubert_labels):
        proj = {'angry':'angry','disgust':'disgust','fear':'fearful',
                'happy':'happy','neutral':'neutral','sad':'sad'}.get(hlabel)
        if proj and proj in EMOTION_CLASSES:
            project_probs[EMOTION_CLASSES.index(proj)] = hubert_probs[i]

    total = project_probs.sum()
    if total > 0:
        project_probs /= total

    return project_probs


# ── Main ──────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n{"="*55}')
    print(f'  TEST HuBERT — {N_PER_CLASS} clipuri/clasa')
    print(f'  Device: {device}')
    print(f'{"="*55}\n')

    # Incarcare model
    print('[INFO] Incarcare HuBERT...')
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(HUBERT_DIR))
    model = HubertForSpeechClassification.from_pretrained(
        str(HUBERT_DIR)
    ).to(device)
    model.eval()
    hubert_labels = [model.config.id2label[i]
                     for i in range(len(model.config.id2label))]
    print(f'  Clase HuBERT: {hubert_labels}')

    # Subset
    print('\n[INFO] Construire subset echilibrat din test.csv...')
    df = build_subset(PROJECT_DIR / 'test.csv', n_per_class=N_PER_CLASS)

    # Evaluare
    print('\n[INFO] Evaluare...')
    y_true, y_pred = [], []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Predictii'):
        probs = predict_audio(row['audio_path'], feature_extractor,
                              model, hubert_labels, device)
        if probs is None:
            errors += 1
            continue

        true_idx = EMOTION_CLASSES.index(row['emotion'])
        pred_idx = np.argmax(probs)
        y_true.append(true_idx)
        y_pred.append(pred_idx)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    f1w = f1_score(y_true, y_pred, average='weighted')

    print(f'\n{"="*55}')
    print(f'  REZULTATE — HuBERT pe {len(y_true)} clipuri')
    print(f'{"="*55}')
    print(f'  Accuracy:   {acc*100:.2f}%')
    print(f'  F1 macro:   {f1m*100:.2f}%')
    print(f'  F1 weighted:{f1w*100:.2f}%')
    if errors:
        print(f'  Erori:      {errors} clipuri sarite')

    print(f'\n  Classification Report:')
    print(classification_report(y_true, y_pred,
                                target_names=EMOTION_CLASSES, digits=4))

    # Per clasa
    print('  Acuratete per clasa:')
    for i, em in enumerate(EMOTION_CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == i).sum() / mask.sum()
            print(f'    {em:8s}: {cls_acc*100:.1f}%  ({(y_pred[mask]==i).sum()}/{mask.sum()})')

    print(f'\n{"="*55}\n')


if __name__ == '__main__':
    main()
