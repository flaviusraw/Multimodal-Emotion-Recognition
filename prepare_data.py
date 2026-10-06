"""
prepare_data.py — Construieste CSV-urile doar din CREMA-D
Stocheaza CILE ABSOLUTE catre fisierele video (.flv) si audio (.wav).
NU extrage cadre JPG — citirea video se face la runtime in Dataset.
Clasa 'sad' este exclusa intentionat → 5 emotii.

Split within-actor: fiecare actor contribuie clipuri in TOATE cele 3 seturi.
  75% train / 15% val / 10% test — stratificat pe emotie per actor.
"""
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

from config import *


def collect_crema_clips():
    """
    Scaneaza CREMA-D si construieste perechi (video_path, audio_path, emotion, actor_id).
    Filename format: {ActorID}_{Sentence}_{Emotion}_{Intensity}
    Ex: 1042_ITH_ANG_XX.flv  /  1042_ITH_ANG_XX.wav
    """
    rows = []

    audio_files = {}
    for f in glob.glob(str(CREMA_AUDIO_DIR / "*.wav")):
        stem = Path(f).stem
        audio_files[stem] = f

    video_patterns = [
        str(CREMA_VIDEO_DIR / "*.flv"),
        str(CREMA_VIDEO_DIR / "*.mp4"),
    ]
    video_paths = []
    for pattern in video_patterns:
        video_paths.extend(glob.glob(pattern))

    print(f"  Fisiere video gasite: {len(video_paths)}")
    print(f"  Fisiere audio gasite: {len(audio_files)}")

    if len(video_paths) == 0:
        alt_patterns = [
            str(CREMA_DIR / "*.flv"),
            str(CREMA_DIR / "*.mp4"),
            str(CREMA_DIR / "**" / "*.flv"),
            str(CREMA_DIR / "**" / "*.mp4"),
        ]
        for pattern in alt_patterns:
            video_paths.extend(glob.glob(pattern, recursive=True))
        video_paths = list(set(video_paths))
        if video_paths:
            print(f"  (gasite in alt subdirector: {len(video_paths)})")

    for vpath in video_paths:
        stem = Path(vpath).stem
        parts = stem.split('_')
        if len(parts) < 4:
            continue

        actor_id    = parts[0]
        emotion_code = parts[2]
        emotion = CREMA_EMOTION_MAP.get(emotion_code)
        if emotion is None:
            continue

        audio_path = audio_files.get(stem)
        if audio_path is None:
            continue

        rows.append({
            'video_path': vpath,
            'audio_path': audio_path,
            'emotion':    emotion,
            'actor_id':   actor_id,
            'clip_id':    stem,
        })

    return pd.DataFrame(rows)


def split_within_actor(df, train_size=0.75, val_size=0.15, seed=RANDOM_SEED):
    """
    Split within-actor: fiecare actor contribuie clipuri in TOATE cele 3 seturi.
    Stratificat pe emotie → proportiile claselor sunt pastrate per actor.

    Exemplu: actor cu 20 clipuri/clasa × 5 clase = 100 clipuri
      Train: ~15/clasa × 5 = ~75 clipuri  (75%)
      Val:    ~3/clasa × 5 = ~15 clipuri  (15%)
      Test:   ~2/clasa × 5 = ~10 clipuri  (10%)
    """
    test_size = round(1.0 - train_size - val_size, 10)  # 0.10
    val_frac  = val_size / (train_size + val_size)       # 0.15/0.90 ≈ 0.1667

    train_rows, val_rows, test_rows = [], [], []

    for actor_id, actor_df in df.groupby('actor_id'):
        actor_df = actor_df.reset_index(drop=True)

        # Pasul 1: test (10%) vs train+val (90%)
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                          random_state=seed)
        try:
            trainval_idx, test_idx = next(
                sss_test.split(actor_df, actor_df['emotion'])
            )
        except ValueError:
            # Prea putine clipuri pentru stratificare → tot in train
            train_rows.append(actor_df)
            continue

        trainval_df = actor_df.iloc[trainval_idx].reset_index(drop=True)
        test_rows.append(actor_df.iloc[test_idx])

        # Pasul 2: val (≈16.67% din trainval) vs train
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac,
                                         random_state=seed)
        try:
            train_idx, val_idx = next(
                sss_val.split(trainval_df, trainval_df['emotion'])
            )
        except ValueError:
            train_rows.append(trainval_df)
            continue

        train_rows.append(trainval_df.iloc[train_idx])
        val_rows.append(trainval_df.iloc[val_idx])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df   = pd.concat(val_rows,   ignore_index=True)
    test_df  = pd.concat(test_rows,  ignore_index=True)

    return train_df, val_df, test_df


def main():
    print("=" * 70)
    print("PREPARARE DATE — CREMA-D (doar video + audio, fara JPG)")
    print("=" * 70)

    print(f"\nDirectoare CREMA-D:")
    print(f"  Video: {CREMA_VIDEO_DIR}")
    print(f"  Audio: {CREMA_AUDIO_DIR}")

    df = collect_crema_clips()

    if len(df) == 0:
        print("\nERROR: Nu au fost gasite perechi video+audio!")
        print(f"  Video dir: {CREMA_VIDEO_DIR}")
        print(f"  Audio dir: {CREMA_AUDIO_DIR}")
        return

    print(f"\n  Total perechi video+audio: {len(df)}")
    print(f"  Actori unici: {df['actor_id'].nunique()}")
    print(f"  Distributie emotii:")
    print(df['emotion'].value_counts().to_string())

    # ── Split within-actor ───────────────────────────────────────────────
    train_df, val_df, test_df = split_within_actor(df)

    total = len(df)
    print(f"\n  Split within-actor (75% / 15% / 10%):")
    print(f"    Train: {len(train_df)} clipuri ({len(train_df)/total*100:.1f}%), "
          f"{train_df['actor_id'].nunique()} actori")
    print(f"    Val:   {len(val_df)} clipuri ({len(val_df)/total*100:.1f}%), "
          f"{val_df['actor_id'].nunique()} actori")
    print(f"    Test:  {len(test_df)} clipuri ({len(test_df)/total*100:.1f}%), "
          f"{test_df['actor_id'].nunique()} actori")

    # Verifica ca TOTI actorii apar in toate split-urile
    train_actors = set(train_df['actor_id'])
    val_actors   = set(val_df['actor_id'])
    test_actors  = set(test_df['actor_id'])
    all_actors   = set(df['actor_id'])
    assert train_actors == all_actors, "Unii actori lipsesc din train!"
    assert val_actors   == all_actors, "Unii actori lipsesc din val!"
    assert test_actors  == all_actors, "Unii actori lipsesc din test!"
    print("  ✓ Toti actorii prezenti in toate cele 3 split-uri")

    # Distributie emotii per split
    print(f"\n  Distributie emotii per split:")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split['emotion'].value_counts().to_dict()
        print(f"    {name}: { {k: counts.get(k,0) for k in EMOTION_CLASSES} }")

    # ── Salvare CSV-uri ──────────────────────────────────────────────────
    df.to_csv(PROJECT_DIR / "crema_all.csv", index=False)
    train_df.to_csv(PROJECT_DIR / "train.csv", index=False)
    val_df.to_csv(PROJECT_DIR / "val.csv",     index=False)
    test_df.to_csv(PROJECT_DIR / "test.csv",   index=False)

    print(f"\n  CSV-uri salvate in: {PROJECT_DIR}")
    print(f"    crema_all.csv  ({len(df)} clipuri)")
    print(f"    train.csv      ({len(train_df)} clipuri)")
    print(f"    val.csv        ({len(val_df)} clipuri)")
    print(f"    test.csv       ({len(test_df)} clipuri)")

    print(f"\n  Sample-uri din train:")
    for _, row in train_df.head(3).iterrows():
        print(f"    {row['clip_id']}  emotion={row['emotion']}  actor={row['actor_id']}")
        print(f"      video: {row['video_path']}")
        print(f"      audio: {row['audio_path']}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
