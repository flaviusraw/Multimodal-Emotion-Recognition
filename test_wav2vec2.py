"""
test_wav2vec2.py — Evalueaza modelul Wav2Vec2 fine-tuned pe CREMA-D
Model: Yassmen/Wav2Vec2_Fine_tuned_on_CremaD_Speech_Emotion_Recognition

Utilizare:
  # Test pe CREMA-D:
  python test_wav2vec2.py --dataset crema --crema-audio /path/to/crema/AudioWAV

  # Test pe RAVDESS:
  python test_wav2vec2.py --dataset ravdess --ravdess /path/to/ravdess_audio

  # Test pe SAVEE:
  python test_wav2vec2.py --dataset savee --savee /path/to/savee/ALL

  # Test pe TESS:
  python test_wav2vec2.py --dataset tess --tess /path/to/tess

  # Test pe toate:
  python test_wav2vec2.py --dataset all \
      --crema-audio /export/home/acs/stud/f/flavius.rau/crema/AudioWAV \
      --ravdess     /export/home/acs/stud/f/flavius.rau/ravdess_audio \
      --savee       /export/home/acs/stud/f/flavius.rau/savee/ALL \
      --tess        /export/home/acs/stud/f/flavius.rau/tess
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "Yassmen/Wav2Vec2_Fine_tuned_on_CremaD_Speech_Emotion_Recognition"
SR = 16000  # Wav2Vec2 lucreaza la 16kHz

# ── Mapari etichete ───────────────────────────────────────────────────────────

# CREMA-D: codul din filename → eticheta model
CREMA_CODE_MAP = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad',
}

# RAVDESS: codul numeric din filename → eticheta model
# Format: 03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
# Emotii: 01=neutral,02=calm,03=happy,04=sad,05=angry,06=fearful,07=disgust,08=surprised
RAVDESS_CODE_MAP = {
    '01': 'neutral',
    '02': None,        # calm — nu exista in modelul CREMA
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': None,        # surprised — nu exista in modelul CREMA
}

# SAVEE: prefix litera → eticheta model
# Format: {speaker}_{emotion}{number}.wav  ex: DC_a01.wav
SAVEE_CODE_MAP = {
    'a':  'angry',
    'd':  'disgust',
    'f':  'fear',
    'h':  'happy',
    'n':  'neutral',
    'sa': 'sad',
    'su': None,  # surprised
}

# TESS: suffix din folder/fisier → eticheta model
# Format: OAF_{word}_{emotion}.wav sau YAF_{word}_{emotion}.wav
TESS_CODE_MAP = {
    'angry':    'angry',
    'disgust':  'disgust',
    'fear':     'fear',
    'happy':    'happy',
    'neutral':  'neutral',
    'sad':      'sad',
    'ps':       None,   # pleasant surprise
}


def load_model():
    print(f"[INFO] Incarcare model: {MODEL_ID}")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model     = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = model.to(device)
    print(f"[INFO] Model incarcat pe {device}")
    print(f"[INFO] Clase model: {model.config.id2label}")
    return model, extractor, device


def predict_file(path, model, extractor, device):
    """Incarca un fisier .wav si returneaza eticheta prezisa si probabilitatile."""
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        return None, None

    inputs = extractor(y, sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs   = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = np.argmax(probs)
    pred    = model.config.id2label[pred_id]
    return pred.lower(), probs


def collect_crema(audio_dir):
    """Colecteaza fisiere CREMA-D cu etichetele adevarate."""
    files  = sorted(glob.glob(str(Path(audio_dir) / "*.wav")))
    data   = []
    for f in files:
        stem  = Path(f).stem          # ex: 1042_ITH_ANG_XX
        parts = stem.split('_')
        if len(parts) < 3:
            continue
        code  = parts[2]
        label = CREMA_CODE_MAP.get(code)
        if label:
            data.append((f, label))
    print(f"[CREMA-D] {len(data)} fisiere valide")
    return data


def collect_ravdess(audio_dir):
    """Colecteaza fisiere RAVDESS cu etichetele adevarate."""
    files = sorted(glob.glob(str(Path(audio_dir) / "**" / "*.wav"), recursive=True))
    data  = []
    for f in files:
        name  = Path(f).stem          # ex: 03-01-05-01-01-01-01
        parts = name.split('-')
        if len(parts) < 3:
            continue
        # Doar speech (modalitate 03), nu song
        if parts[0] != '03':
            continue
        code  = parts[2]
        label = RAVDESS_CODE_MAP.get(code)
        if label:
            data.append((f, label))
    print(f"[RAVDESS] {len(data)} fisiere valide (speech only, fara calm/surprised)")
    return data


def collect_savee(audio_dir):
    """Colecteaza fisiere SAVEE cu etichetele adevarate."""
    files = sorted(glob.glob(str(Path(audio_dir) / "*.wav")))
    data  = []
    for f in files:
        name = Path(f).stem.lower()   # ex: dc_a01
        # Extrage codul emotiei (literele de dupa _)
        m = re.match(r'^[a-z]+_([a-z]+)\d+$', name)
        if not m:
            continue
        code  = m.group(1)
        label = SAVEE_CODE_MAP.get(code)
        if label:
            data.append((f, label))
    print(f"[SAVEE] {len(data)} fisiere valide")
    return data


def collect_tess(audio_dir):
    """Colecteaza fisiere TESS cu etichetele adevarate."""
    files = sorted(glob.glob(str(Path(audio_dir) / "**" / "*.wav"), recursive=True))
    data  = []
    for f in files:
        name  = Path(f).stem.lower()  # ex: oaf_back_angry
        parts = name.split('_')
        if len(parts) < 2:
            continue
        code  = parts[-1]
        label = TESS_CODE_MAP.get(code)
        if label:
            data.append((f, label))
    print(f"[TESS] {len(data)} fisiere valide")
    return data


def evaluate(data, model, extractor, device, dataset_name):
    """Ruleaza inferenta si afiseaza metricile."""
    if not data:
        print(f"[WARN] Niciun fisier pentru {dataset_name}")
        return

    y_true, y_pred = [], []

    for path, true_label in tqdm(data, desc=f"  Evaluare {dataset_name}"):
        pred, _ = predict_file(path, model, extractor, device)
        if pred is None:
            continue
        y_true.append(true_label)
        y_pred.append(pred)

    if not y_true:
        print(f"[WARN] Nu s-au putut procesa fisiere pentru {dataset_name}")
        return

    acc    = accuracy_score(y_true, y_pred)
    labels = sorted(set(y_true))

    print(f"\n{'='*60}")
    print(f"  {dataset_name} — Rezultate ({len(y_true)} fisiere)")
    print(f"  Acuratete: {acc*100:.2f}%")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, labels=labels, digits=4))

    # Matrice de confuzie
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("  Matrice de confuzie (randuri=real, coloane=prezis):")
    header = f"{'':12s}" + "".join(f"{l[:6]:>8s}" for l in labels)
    print(f"  {header}")
    for i, row_label in enumerate(labels):
        row = "".join(f"{cm[i,j]:>8d}" for j in range(len(labels)))
        print(f"  {row_label:12s}{row}")
    print()

    return acc, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(
        description='Evalueaza Wav2Vec2 pe dataset-uri de emotii'
    )
    parser.add_argument('--dataset', choices=['crema', 'ravdess', 'savee', 'tess', 'all'],
                        default='crema', help='Dataset de evaluat')
    parser.add_argument('--crema-audio', type=str,
                        default='/export/home/acs/stud/f/flavius.rau/crema/AudioWAV')
    parser.add_argument('--ravdess',     type=str,
                        default='/export/home/acs/stud/f/flavius.rau/ravdess_audio')
    parser.add_argument('--savee',       type=str,
                        default='/export/home/acs/stud/f/flavius.rau/savee/ALL')
    parser.add_argument('--tess',        type=str,
                        default='/export/home/acs/stud/f/flavius.rau/tess')
    parser.add_argument('--limit',       type=int, default=None,
                        help='Limiteaza nr. de fisiere per dataset (pentru test rapid)')
    args = parser.parse_args()

    model, extractor, device = load_model()

    results = {}
    run = args.dataset

    if run in ('crema', 'all'):
        data = collect_crema(args.crema_audio)
        if args.limit:
            data = data[:args.limit]
        r = evaluate(data, model, extractor, device, 'CREMA-D')
        if r:
            results['CREMA-D'] = r[0]

    if run in ('ravdess', 'all'):
        data = collect_ravdess(args.ravdess)
        if args.limit:
            data = data[:args.limit]
        r = evaluate(data, model, extractor, device, 'RAVDESS')
        if r:
            results['RAVDESS'] = r[0]

    if run in ('savee', 'all'):
        data = collect_savee(args.savee)
        if args.limit:
            data = data[:args.limit]
        r = evaluate(data, model, extractor, device, 'SAVEE')
        if r:
            results['SAVEE'] = r[0]

    if run in ('tess', 'all'):
        data = collect_tess(args.tess)
        if args.limit:
            data = data[:args.limit]
        r = evaluate(data, model, extractor, device, 'TESS')
        if r:
            results['TESS'] = r[0]

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  SUMAR FINAL")
        print(f"{'='*60}")
        for ds, acc in results.items():
            print(f"  {ds:12s}: {acc*100:.2f}%")
        print()


if __name__ == '__main__':
    main()
