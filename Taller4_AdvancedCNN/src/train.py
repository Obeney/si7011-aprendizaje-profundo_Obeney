"""Trainer factory y curvas de entrenamiento."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

from .utils import BATCH_SIZE, EPOCHS, SEED


def make_trainer(
    model,
    tokenizer,
    tokenized_ds,
    output_dir: str,
    compute_metrics,
    lr: float = 2e-5,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    n_train_steps = (len(tokenized_ds["train"]) // batch_size) * epochs
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=int(0.1 * n_train_steps),
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )


def plot_training_curves(trainer, title: str = ""):
    logs = pd.DataFrame(trainer.state.log_history)
    train_logs = logs[logs["loss"].notna()][["step", "loss", "learning_rate"]]
    eval_logs = logs[logs["eval_loss"].notna()][
        ["epoch", "eval_loss", "eval_f1_macro", "eval_accuracy"]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(train_logs["step"], train_logs["loss"], color="steelblue", linewidth=1.5)
    axes[0].set_title("Training Loss", fontweight="bold")

    axes[1].plot(
        eval_logs["epoch"], eval_logs["eval_f1_macro"], marker="o", color="#e74c3c", label="F1 macro"
    )
    axes[1].plot(
        eval_logs["epoch"], eval_logs["eval_accuracy"], marker="s", color="#2ecc71", label="Accuracy"
    )
    axes[1].set_title("Métricas de Validación", fontweight="bold")
    axes[1].set_ylim(0.3, 1.0)
    axes[1].legend()

    axes[2].plot(train_logs["step"], train_logs["learning_rate"], color="purple", linewidth=1.5)
    axes[2].set_title("Learning Rate Schedule", fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
