"""
train_utils.py — Utilitati pentru antrenare
  - Focal Loss + Label Smoothing
  - Warmup + Cosine Annealing scheduler
  - MixUp
  - Evaluare cu metrici complete
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from config import EMOTION_CLASSES, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS cu Label Smoothing
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    Focal Loss: reduce pierderea pe sample-urile usor clasificate,
    focalizand antrenarea pe cele dificile.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.1,
                 weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.size(1)

        # Label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth = F.one_hot(targets, num_classes).float()

        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)

        # Focal weight
        focal_weight = (1 - prob) ** self.gamma

        loss = -focal_weight * smooth * log_prob

        if self.weight is not None:
            w = self.weight[targets].unsqueeze(1)
            loss = loss * w

        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# WARMUP + COSINE ANNEALING SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════
class WarmupCosineScheduler:
    """Linear warmup urmat de cosine annealing."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ══════════════════════════════════════════════════════════════════════════════
# MIXUP
# ══════════════════════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=0.3, num_classes=NUM_CLASSES):
    """
    MixUp: combina sample-uri cu ponderi random din distributia Beta.
    Returneaza: (mixed_x, mixed_y_onehot)
    """
    if alpha <= 0:
        y_oh = F.one_hot(y, num_classes).float()
        return x, y_oh

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Asigura ca lambda >= 0.5

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_oh = F.one_hot(y, num_classes).float()
    mixed_y = lam * y_oh + (1 - lam) * y_oh[index]

    return mixed_x, mixed_y


def mixup_criterion(logits, mixed_targets):
    """Cross-entropy cu targets soft (de la MixUp)."""
    log_prob = F.log_softmax(logits, dim=1)
    return -(mixed_targets * log_prob).sum(dim=1).mean()


# ══════════════════════════════════════════════════════════════════════════════
# EVALUARE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_predictions(y_true, y_pred, class_names=EMOTION_CLASSES):
    """Calculeaza metrici complete."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'macro_precision': prec,
        'macro_recall': rec,
        'macro_f1': f1,
        'report': report,
        'confusion_matrix': cm,
    }


def print_metrics(metrics, title=""):
    """Printeaza frumos metricile."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}  ({metrics['macro_precision']*100:.2f}%)")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}  ({metrics['macro_recall']*100:.2f}%)")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}  ({metrics['macro_f1']*100:.2f}%)")
    print(f"\n{metrics['report']}")
