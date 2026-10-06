"""
evaluate.py — Evaluare completa cu vizualizari
Genereaza:
  - Confusion matrices pentru toate cele 3 modele
  - Grafice comparative
  - Raport complet de clasificare
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

from config import EMOTION_CLASSES, RESULTS_DIR


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Deseneaza si salveaza confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Valori absolute
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title(f'{title} — Counts')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Procente
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title(f'{title} — Normalized')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_bar(models_metrics, save_path):
    """Grafic bar comparativ pentru toate modelele."""
    metrics_names = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
    model_names = list(models_metrics.keys())
    
    x = np.arange(len(metrics_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2196F3', '#FF9800', '#4CAF50']
    for i, (model_name, m) in enumerate(models_metrics.items()):
        values = [m['accuracy'] * 100, m['macro_precision'] * 100,
                  m['macro_recall'] * 100, m['macro_f1'] * 100]
        bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score (%)')
    ax.set_title('Comparatie Performante — Video vs Audio vs Fusion')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_f1(models_metrics, class_names, save_path):
    """Grafic per-class F1 pentru fiecare model."""
    model_names = list(models_metrics.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for i, (model_name, m) in enumerate(models_metrics.items()):
        if 'per_class_f1' in m:
            ax.bar(x + i * width, [v * 100 for v in m['per_class_f1']],
                   width, label=model_name, color=colors[i])

    ax.set_ylabel('F1 Score (%)')
    ax.set_title('F1 Score per Emotie')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def compute_metrics_from_arrays(y_true, y_pred, class_names):
    """Calculeaza metrici complete."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        precision_recall_fscore_support,
    )
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class
    _, _, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    report = classification_report(y_true, y_pred,
                                   target_names=class_names, digits=4)
    return {
        'accuracy': acc,
        'macro_precision': prec,
        'macro_recall': rec,
        'macro_f1': f1,
        'per_class_f1': f1_per.tolist(),
        'report': report,
    }


def main():
    print("=" * 60)
    print("EVALUARE COMPLETA — Multimodal Emotion Recognition")
    print("=" * 60)

    results = {}

    # ── Incarca predictii salvate ────────────────────────────────────────
    model_files = {
        'Video (ViT)': ('video_test_preds.npy', 'video_test_labels.npy'),
        'Audio (Transformer)': ('audio_test_preds.npy', 'audio_test_labels.npy'),
        'Fusion (Weighted)': ('fusion_test_preds.npy', 'fusion_test_labels.npy'),
    }

    for model_name, (pred_file, label_file) in model_files.items():
        pred_path = RESULTS_DIR / pred_file
        label_path = RESULTS_DIR / label_file

        if not pred_path.exists() or not label_path.exists():
            print(f"\n  ⚠ {model_name}: fisierele nu exista, skip.")
            continue

        preds = np.load(pred_path)
        labels = np.load(label_path)

        metrics = compute_metrics_from_arrays(labels, preds, EMOTION_CLASSES)
        results[model_name] = metrics

        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")
        print(f"  Accuracy:        {metrics['accuracy']*100:.2f}%")
        print(f"  Macro Precision: {metrics['macro_precision']*100:.2f}%")
        print(f"  Macro Recall:    {metrics['macro_recall']*100:.2f}%")
        print(f"  Macro F1:        {metrics['macro_f1']*100:.2f}%")
        print(f"\n{metrics['report']}")

        # Confusion matrix
        cm_path = RESULTS_DIR / f"cm_{pred_file.replace('_test_preds.npy', '')}.png"
        plot_confusion_matrix(labels, preds, EMOTION_CLASSES, model_name, cm_path)

    # ── Grafice comparative ──────────────────────────────────────────────
    if len(results) >= 2:
        print("\nGenerare grafice comparative...")
        plot_comparison_bar(results, RESULTS_DIR / "comparison_bar.png")
        plot_per_class_f1(results, EMOTION_CLASSES, RESULTS_DIR / "per_class_f1.png")

    # ── Rezumat final ────────────────────────────────────────────────────
    if results:
        print(f"\n{'='*60}")
        print("REZUMAT FINAL")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print(f"{'-'*57}")
        for name, m in results.items():
            print(f"{name:<25} {m['accuracy']*100:>7.2f}% "
                  f"{m['macro_precision']*100:>7.2f}% "
                  f"{m['macro_recall']*100:>7.2f}% "
                  f"{m['macro_f1']*100:>7.2f}%")
        print(f"{'='*60}")

    print(f"\nToate rezultatele salvate in: {RESULTS_DIR}")
    print("DONE!")


if __name__ == '__main__':
    main()
