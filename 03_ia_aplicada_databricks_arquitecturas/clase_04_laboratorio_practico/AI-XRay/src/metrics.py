"""
Evaluación del modelo más allá de accuracy: matriz de confusión, precision, recall,
F1, ROC-AUC y classification report -- clave en un problema médico desbalanceado,
donde un modelo que siempre predice "PNEUMONIA" puede tener accuracy alta y ser inútil
(o peligroso) en la práctica.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin display, para correr en notebooks/servidores headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import CLASSES


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Calcula el set completo de métricas a partir de las probabilidades predichas.
    Todas las métricas quedan en un dict plano, listo para `mlflow.log_metrics(...)`.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def classification_report_text(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> str:
    y_pred = (y_prob >= threshold).astype(int)
    return classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)


def save_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, threshold: float = 0.5) -> Path:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, name="AI-XRay", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC=0.50)")
    ax.set_title("Curva ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_training_curves(history, out_path: Path) -> Path:
    """Grafica loss/accuracy de entrenamiento vs. validación a partir del `history`
    que devuelve `model.fit(...)`."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Época")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
