"""
Arquitectura del modelo: ResNet50 preentrenada en ImageNet + Transfer Learning.

Pipeline:  Imagen -> ResNet50 (conv base) -> GlobalAveragePooling2D -> Dropout -> Dense(1, sigmoid)

Diseñado deliberadamente para clasificación BINARIA (NORMAL vs PNEUMONIA) en esta v1,
pero con `num_classes` como parámetro: para ampliar a multiclase (por ejemplo, separar
PNEUMONIA en bacteriana/viral, algo que el propio nombre de archivo del dataset ya
permite) basta con llamar `build_model(num_classes=3)`, que cambia automáticamente la
capa de salida a `Dense(num_classes, activation="softmax")` y la loss recomendada a
`categorical_crossentropy`.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50

from src.config import IMG_CHANNELS, IMG_SIZE, SEED


def set_seeds(seed: int = SEED) -> None:
    """Fija semillas para reproducibilidad razonable (no 100% determinista en GPU,
    pero suficiente para que las comparaciones entre runs de MLflow sean justas)."""
    tf.keras.utils.set_random_seed(seed)


def build_model(
    dropout: float = 0.3,
    num_classes: int = 1,
    base_trainable: bool = False,
    fine_tune_at: int = None,
    weights: str = "imagenet",
) -> tf.keras.Model:
    """Construye el modelo de clasificación.

    Args:
        dropout: tasa de Dropout antes de la capa densa final.
        num_classes: 1 para binario (sigmoid), >1 para multiclase (softmax).
        base_trainable: si False, TODA la base de ResNet50 queda congelada (Fase 1
            del entrenamiento: solo se entrena el clasificador nuevo).
        fine_tune_at: si `base_trainable=True`, índice de capa a partir del cual se
            descongela la base (Fase 2: fine-tuning de las últimas capas). Las capas
        weights: "imagenet" (por defecto, requiere descargar los pesos preentrenados la
            primera vez) o `None` para inicializar la base al azar -- útil solo para
            pruebas rápidas del pipeline sin conexión a internet, nunca para el modelo
            final (perdería toda la ventaja del transfer learning).
            anteriores a este índice permanecen congeladas para no destruir los
            features genéricos aprendidos en ImageNet con pocas imágenes de rayos X.
    """
    base_model = ResNet50(
        weights=weights,
        include_top=False,
        input_shape=(*IMG_SIZE, IMG_CHANNELS),
    )

    if not base_trainable:
        base_model.trainable = False
    else:
        base_model.trainable = True
        if fine_tune_at is not None:
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, IMG_CHANNELS))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    model = models.Model(inputs, outputs, name="aixray_resnet50")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float, num_classes: int = 1) -> None:
    loss = "binary_crossentropy" if num_classes == 1 else "categorical_crossentropy"
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
