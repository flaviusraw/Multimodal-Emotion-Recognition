import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from config import BASE_DIR, PROJECT_DIR, CREMA_EMOTION_MAP, RANDOM_SEED

VIDEO_DIR = BASE_DIR / "crema" / "VideoFlash"

rows = []
# Cautam fisierele .flv
for f in VIDEO_DIR.glob("*.flv"):
    fname = f.stem
    parts = fname.split('_')
    if len(parts) < 3:
        continue
    emotion = CREMA_EMOTION_MAP.get(parts[2])
    if not emotion:
        continue
    actor_id = f"crema_{parts[0]}"
    rows.append({
        'video_path': str(f), 
        'emotion': emotion, 
        'actor_id': actor_id
    })

df = pd.DataFrame(rows)
print(f"Am gasit {len(df)} videoclipuri .flv")

# Split: 15% Test
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
trainval_idx, test_idx = next(gss_test.split(df, df['emotion'], groups=df['actor_id']))
trainval_df = df.iloc[trainval_idx].copy()
test_df = df.iloc[test_idx].copy()

# Split: 10% Val din restul
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_SEED)
train_idx, val_idx = next(gss_val.split(trainval_df, trainval_df['emotion'], groups=trainval_df['actor_id']))
train_df = trainval_df.iloc[train_idx].copy()
val_df = trainval_df.iloc[val_idx].copy()

train_df.to_csv(PROJECT_DIR / "video_train.csv", index=False)
val_df.to_csv(PROJECT_DIR / "video_val.csv", index=False)
test_df.to_csv(PROJECT_DIR / "video_test.csv", index=False)

print(f"Salvat: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
