"""
Pipeline de `tf.data` para cargar, decodificar, preprocesar y (opcionalmente) aumentar
las imágenes de rayos X, a partir del manifiesto construido en `src/dataset.py`.

Punto importante: ResNet50 con pesos de ImageNet espera entradas 224x224x3 preprocesadas
con `tensorflow.keras.applications.resnet50.preprocess_input` (que centra los canales
según las estadísticas de ImageNet), NO una normalización genérica a [0, 1]. Usar la
normalización equivocada es un error común que degrada silenciosamente el rendimiento
del *transfer learning*.
"""

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.config import BATCH_SIZE, IMG_SIZE, SEED

AUTOTUNE = tf.data.AUTOTUNE

# Augmentation solo para el set de entrenamiento. Se implementa como capas de Keras
# (en vez de operaciones sueltas de tf.image) para que viajen DENTRO del grafo del
# modelo si se quisiera exportarlas junto con él; aquí se aplican en el pipeline de datos.
_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.05, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
        tf.keras.layers.RandomContrast(0.1, seed=SEED),
    ],
    name="augmentation",
)


def _decode_and_resize(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img


def build_dataset(
    paths, labels, split: str, batch_size: int = BATCH_SIZE, shuffle_buffer: int = 1024
) -> tf.data.Dataset:
    """Construye un `tf.data.Dataset` listo para `model.fit` / `model.evaluate`.

    - `split="train"`: aplica shuffle + data augmentation.
    - `split in ("val", "test")`: sin augmentation ni shuffle (evaluación determinista).
    """
    paths = tf.constant(list(paths))
    labels = tf.constant(list(labels), dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if split == "train":
        ds = ds.shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _decode_and_resize(path)
        if split == "train":
            img = _augmentation(img, training=True)
        img = preprocess_input(img)
        return img, label

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def dataset_from_manifest(manifest_df, split: str, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Atajo: filtra el manifiesto por la columna `split` y arma el `tf.data.Dataset`."""
    subset = manifest_df[manifest_df["split"] == split]
    return build_dataset(subset["path"].tolist(), subset["label"].tolist(), split, batch_size)
