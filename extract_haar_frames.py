"""
extract_haar_frames.py — Extrage 64 frame-uri cu Haar Cascade 224x224
pentru toate clipurile CREMA-D si le salveaza pe disk.

Structura output:
  HAAR_FRAMES_DIR/
    1001_IEO_ANG_LO/
      frame_00.jpg ... frame_63.jpg
    1001_IEO_DIS_LO/
      ...

Utilizare:
  python extract_haar_frames.py
  python extract_haar_frames.py --workers 8   # paralel
  python extract_haar_frames.py --outdir /alt/director
"""
import argparse
import glob
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import CREMA_VIDEO_DIR, CREMA_EMOTION_MAP, PROJECT_DIR

# Director output default
DEFAULT_OUTDIR = PROJECT_DIR / "haar_frames"

# Haar Cascade
HAAR_XML = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

NUM_FRAMES = 64
OUT_SIZE   = 224


def crop_face_224(frame_rgb, face_cascade, cx=None, cy=None):
    """
    Crop 224x224 centrat pe fata.
    Daca cx/cy sunt date, foloseste direct (evita re-detectia per frame).
    """
    h, w = frame_rgb.shape[:2]
    half = OUT_SIZE // 2

    if cx is None or cy is None:
        gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            cx, cy = x + fw // 2, y + fh // 2
        else:
            cx, cy = w // 2, h // 2

    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    pad_top    = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left   = max(0, -x1)
    pad_right  = max(0, x2 - w)

    crop = frame_rgb[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

    if pad_top or pad_bottom or pad_left or pad_right:
        crop = np.pad(crop,
                      ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                      mode='constant', constant_values=0)

    if crop.shape[0] != OUT_SIZE or crop.shape[1] != OUT_SIZE:
        crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE))

    return crop, cx, cy


def process_clip(args):
    """
    Proceseaza un singur clip .flv:
      - Citeste toate frame-urile
      - Detecteaza fata pe frame-ul din mijloc (o singura data)
      - Uniform sampling → 64 frame-uri
      - Crop 224x224 centrat pe fata pentru fiecare frame
      - Salveaza ca JPEG in outdir/clip_id/frame_XX.jpg
    """
    video_path, outdir = args
    stem  = Path(video_path).stem
    parts = stem.split('_')
    if len(parts) < 4:
        return stem, False, "denumire invalida"

    emotion_code = parts[2]
    if emotion_code not in CREMA_EMOTION_MAP:
        return stem, False, f"emotie necunoscuta: {emotion_code}"

    clip_outdir = Path(outdir) / stem
    # Skip daca deja procesat complet
    if clip_outdir.exists() and len(list(clip_outdir.glob("*.jpg"))) == NUM_FRAMES:
        return stem, True, "skip (deja exista)"

    clip_outdir.mkdir(parents=True, exist_ok=True)

    # Citeste frame-urile
    cap    = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        return stem, False, "video gol"

    # Detectie fata pe frame-ul din mijloc
    face_cascade = cv2.CascadeClassifier(HAAR_XML)
    mid = frames[len(frames) // 2]
    _, cx, cy = crop_face_224(mid, face_cascade)

    # Uniform sampling → 64 frame-uri
    total   = len(frames)
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)

    for i, idx in enumerate(indices):
        frame_rgb = frames[idx]
        cropped, _, _ = crop_face_224(frame_rgb, face_cascade, cx=cx, cy=cy)
        out_path = clip_outdir / f"frame_{i:02d}.jpg"
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

    return stem, True, f"{len(frames)} frame-uri sursa"


def main():
    parser = argparse.ArgumentParser(
        description='Extrage 64 frame-uri Haar 224x224 din toate clipurile CREMA-D'
    )
    parser.add_argument('--outdir',  type=str, default=str(DEFAULT_OUTDIR),
                        help=f'Director output (default: {DEFAULT_OUTDIR})')
    parser.add_argument('--workers', type=int, default=4,
                        help='Numar procese paralele (default: 4)')
    parser.add_argument('--limit',   type=int, default=None,
                        help='Proceseaza doar primele N clipuri (pentru test)')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Gaseste toate fisierele video
    video_paths = glob.glob(str(CREMA_VIDEO_DIR / "*.flv"))
    video_paths += glob.glob(str(CREMA_VIDEO_DIR / "*.mp4"))

    if len(video_paths) == 0:
        print(f"[EROARE] Niciun fisier video gasit in: {CREMA_VIDEO_DIR}")
        return

    if args.limit:
        video_paths = video_paths[:args.limit]

    print(f"{'='*60}")
    print(f"  EXTRACTIE HAAR FRAMES — CREMA-D")
    print(f"  Clipuri de procesat: {len(video_paths)}")
    print(f"  Frame-uri per clip:  {NUM_FRAMES}")
    print(f"  Dimensiune:          {OUT_SIZE}x{OUT_SIZE}")
    print(f"  Output:              {outdir}")
    print(f"  Procese paralele:    {args.workers}")
    print(f"{'='*60}\n")

    # Estimeaza spatiu necesar
    # 64 frame-uri × ~15KB JPEG × 7442 clipuri ≈ ~7GB
    space_gb = len(video_paths) * NUM_FRAMES * 15 / 1024 / 1024
    print(f"  Spatiu estimat: ~{space_gb:.1f} GB\n")

    tasks = [(vp, str(outdir)) for vp in video_paths]

    ok_count   = 0
    skip_count = 0
    err_count  = 0
    errors     = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_clip, t): t for t in tasks}
        bar = tqdm(as_completed(futures), total=len(tasks),
                   desc="Procesare clipuri", unit="clip")
        for future in bar:
            stem, success, msg = future.result()
            if success:
                if 'skip' in msg:
                    skip_count += 1
                else:
                    ok_count += 1
            else:
                err_count += 1
                errors.append(f"{stem}: {msg}")
            bar.set_postfix(ok=ok_count, skip=skip_count, err=err_count)

    print(f"\n{'='*60}")
    print(f"  Procesate cu succes: {ok_count}")
    print(f"  Sarite (existente):  {skip_count}")
    print(f"  Erori:               {err_count}")
    if errors:
        print(f"\n  Primele erori:")
        for e in errors[:5]:
            print(f"    {e}")
    print(f"\n  Frame-uri salvate in: {outdir.resolve()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
