"""
Entrenamiento de una variante del modelo + registro completo del experimento en MLflow.

`train_and_log(...)` es la función que los notebooks 03_training.ipynb llaman una vez
por cada variante que se quiere comparar (por ejemplo, los 3 runs de referencia:
lr=0.001/dropout=0.3 congelado, lr=0.0001/dropout=0.5 congelado, y fine-tuning con
lr=0.00001). Cada llamada es un `mlflow.start_run()` independiente y comparable.
"""

import tempfile
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from mlflow.models import infer_signature
from sklearn.utils.class_weight import compute_class_weight

from src.config import IMG_CHANNELS, IMG_SIZE, SEED
from src.metrics import (
    classification_report_text,
    compute_metrics,
    save_confusion_matrix,
    save_roc_curve,
    save_training_curves,
)
from src.architecture import build_model, compile_model, set_seeds


def train_and_log(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    run_name: str,
    learning_rate: float,
    dropout: float,
    epochs: int = 15,
    fine_tune: bool = False,
    fine_tune_at: int = 143,
    class_weight: dict = None,
    tags: dict = None,
    register_if_best: bool = False,
    weights: str = "imagenet",
) -> str:
    """Entrena una variante del modelo dentro de un run de MLflow y registra todo:
    parámetros, métricas por época, métricas finales de evaluación, artefactos
    (curvas de entrenamiento, matriz de confusión, curva ROC, classification report)
    y el propio modelo con su *signature*.

    `weights="imagenet"` por defecto (lo correcto para el laboratorio real). Se puede
    pasar `weights=None` únicamente para probar el pipeline sin descargar pesos
    preentrenados (por ejemplo, en un entorno sin salida a internet) -- el modelo
    resultante no sirve como clasificador real, solo para verificar que el código corre.

    Devuelve el `run_id` de MLflow, para poder comparar/registrar después.
    """
    set_seeds(SEED)

    model = build_model(
        dropout=dropout,
        base_trainable=fine_tune,
        fine_tune_at=fine_tune_at if fine_tune else None,
        weights=weights,
    )
    compile_model(model, learning_rate=learning_rate)

    with mlflow.start_run(run_name=run_name) as run:
        # -- 1. Parámetros --------------------------------------------------------
        mlflow.log_param("architecture", "ResNet50_fine_tuning" if fine_tune else "ResNet50_frozen")
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("epochs_max", epochs)
        mlflow.log_param("batch_size", train_ds.element_spec[0].shape[0] or "dynamic")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("fine_tune", fine_tune)
        mlflow.log_param("fine_tune_at_layer", fine_tune_at if fine_tune else None)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("class_weight", class_weight)
        if tags:
            mlflow.set_tags(tags)

        # -- 2. Callbacks -----------------------------------------------------------
        checkpoint_path = Path(tempfile.mkdtemp()) / f"{run_name}_best.keras"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True
            ),
        ]

        # -- 3. Entrenamiento ---------------------------------------------------
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )

        # MLflow no registra automáticamente el histórico por época con este flujo manual
        # (a diferencia de mlflow.tensorflow.autolog()), así que se loguea explícitamente.
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(
            zip(
                history.history["loss"],
                history.history["accuracy"],
                history.history["val_loss"],
                history.history["val_accuracy"],
            )
        ):
            mlflow.log_metrics(
                {"loss": loss, "accuracy": acc, "val_loss": val_loss, "val_accuracy": val_acc},
                step=epoch,
            )

        # -- 4. Evaluación completa (no solo accuracy) -----------------------------
        y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
        y_prob = model.predict(val_ds, verbose=0).ravel()

        final_metrics = compute_metrics(y_true, y_prob)
        mlflow.log_metrics(final_metrics)
        print(classification_report_text(y_true, y_prob))

        tmp_dir = Path(tempfile.mkdtemp())
        cm_path = save_confusion_matrix(y_true, y_prob, tmp_dir / "confusion_matrix.png")
        roc_path = save_roc_curve(y_true, y_prob, tmp_dir / "roc_curve.png")
        curves_path = save_training_curves(history, tmp_dir / "training_curves.png")
        report_path = tmp_dir / "classification_report.txt"
        report_path.write_text(classification_report_text(y_true, y_prob))

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(curves_path))
        mlflow.log_artifact(str(report_path))

        # -- 5. El modelo, con signature -------------------------------------------
        sample_input = np.zeros((1, *IMG_SIZE, IMG_CHANNELS), dtype="float32")
        signature = infer_signature(sample_input, model.predict(sample_input, verbose=0))

        # NOTA / workaround verificado: `mlflow.keras.log_model(..., input_example=...)`
        # falla con FileNotFoundError en la combinación mlflow>=3.x + keras 3 (bug de
        # serialización del input_example para el flavor de keras). Se omite
        # `input_example` en log_model y se deja constancia del shape esperado como un
        # artefacto JSON aparte -- el efecto práctico (documentar la forma de entrada)
        # es el mismo, sin depender del código con el bug.
        mlflow.keras.log_model(model, name="model", signature=signature)
        mlflow.log_dict(
            {"input_shape": list(sample_input.shape), "dtype": "float32", "preprocess": "resnet50.preprocess_input"},
            "input_example_info.json",
        )

        run_id = run.info.run_id

    print(f"Run '{run_name}' completo -- run_id={run_id} -- métricas: {final_metrics}")
    return run_id


def compute_balanced_class_weight(labels) -> dict:
    """Calcula `class_weight` (dict {0: peso, 1: peso}) para compensar desbalance de
    clases. En el subconjunto balanceado de la demo de clase esto da pesos ~1.0/1.0
    (no hace nada); su utilidad real aparece al entrenar con el dataset completo
    (desbalanceado ~3:1 a favor de PNEUMONIA) -- ver README, sección de extensión."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}
