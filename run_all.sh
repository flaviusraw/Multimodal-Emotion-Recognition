#!/bin/bash
# run_all.sh — Pipeline complet de antrenare
# Ruleaza pe server: bash run_all.sh

set -e

echo "============================================================"
echo " MULTIMODAL EMOTION RECOGNITION — Pipeline Complet"
echo "============================================================"

cd "$(dirname "$0")"

# Etapa 1: Pregatire date
echo ""
echo "[1/5] Pregatire date..."
python prepare_data.py

# Etapa 2: Antrenare model video (ViT)
echo ""
echo "[2/5] Antrenare Video ViT..."
python train_video.py

# Etapa 3: Antrenare model audio (Transformer)
echo ""
echo "[3/5] Antrenare Audio Transformer..."
python train_audio.py

# Etapa 4: Antrenare fuziune
echo ""
echo "[4/5] Antrenare Weighted Late Fusion..."
python train_fusion.py

# Etapa 5: Evaluare completa
echo ""
echo "[5/5] Evaluare finala..."
python evaluate.py

echo ""
echo "============================================================"
echo " PIPELINE COMPLET!"
echo " Rezultate in: multimodal_emotion/results/"
echo " Modele in:    multimodal_emotion/checkpoints/"
echo "============================================================"
