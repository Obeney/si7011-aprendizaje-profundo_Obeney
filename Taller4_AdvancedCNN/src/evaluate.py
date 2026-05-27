"""Métricas y evaluación visual."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from .utils import LABEL_NAMES


def compute_metrics_factory(f1_metric, acc_metric):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
        acc = acc_metric.compute(predictions=preds, references=labels)
        return {"f1_macro": f1["f1"], "accuracy": acc["accuracy"]}

    return compute_metrics


def full_evaluation(trainer, test_dataset, model_name: str = ""):
    output = trainer.predict(test_dataset)
    preds = np.argmax(output.predictions, axis=1)
    labels = output.label_ids

    title = f"Test — {model_name}" if model_name else "Test"
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")
    print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=4))

    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(
        ax=axes[0], colorbar=False, cmap="Blues", xticks_rotation=30
    )
    axes[0].set_title("Confusion Matrix (conteos)", fontweight="bold")
    ConfusionMatrixDisplay(cm_norm, display_labels=LABEL_NAMES).plot(
        ax=axes[1], colorbar=False, cmap="Blues", values_format=".2f", xticks_rotation=30
    )
    axes[1].set_title("Confusion Matrix (normalizada)", fontweight="bold")
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return output.metrics
