"""
Optimiza el modelo base entrenado en train_model.py aplicando, en orden:

1. Pruning (poda gradual de magnitud) con tensorflow-model-optimization,
   seguido de fine-tuning corto para recuperar precisión.
2. Quantization dinámica al convertir a TensorFlow Lite.

Genera dos artefactos nuevos en la carpeta models/:
    - modelo_podado.h5        (mismo formato, pesos podados, sin cuantizar)
    - modelo_optimizado.tflite (podado + cuantizado, listo para servir)

Uso:
    python src/optimize_model.py
"""
import os

# Debe fijarse ANTES de importar tensorflow/tensorflow_model_optimization.
# Ver la nota equivalente en train_model.py.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BASE_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_base.h5")
PRUNED_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_podado.h5")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_optimizado.tflite")


def cargar_datos():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return (x_train, y_train), (x_test, y_test)


def aplicar_pruning(modelo_base, x_train, y_train, x_test, y_test):
    """Aplica poda gradual de magnitud y hace fine-tuning corto."""
    print("\n[1/2] Aplicando pruning...")

    num_muestras = x_train.shape[0]
    epocas_finetune = 2
    batch_size = 128
    end_step = int(np.ceil(num_muestras / batch_size)) * epocas_finetune

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=end_step,
        )
    }

    modelo_podado = tfmot.sparsity.keras.prune_low_magnitude(modelo_base, **pruning_params)
    modelo_podado.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    modelo_podado.fit(
        x_train, y_train,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=epocas_finetune,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = modelo_podado.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy en test (modelo podado, antes de strip): {test_acc:.4f}")

    # strip_pruning quita las envolturas de pruning y deja un modelo Keras normal,
    # pero con muchos pesos exactamente en cero (listo para comprimir/cuantizar).
    modelo_final = tfmot.sparsity.keras.strip_pruning(modelo_podado)
    modelo_final.save(PRUNED_MODEL_PATH)
    print(f"Modelo podado guardado en: {PRUNED_MODEL_PATH}")

    return modelo_final


def convertir_a_tflite(modelo, x_calibracion):
    """Convierte el modelo a TFLite con quantization dinámica (post-training)."""
    print("\n[2/2] Convirtiendo a TFLite con quantization dinámica...")

    converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Modelo optimizado (.tflite) guardado en: {TFLITE_MODEL_PATH}")


def main():
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró {BASE_MODEL_PATH}. Ejecuta primero: python src/train_model.py"
        )

    (x_train, y_train), (x_test, y_test) = cargar_datos()

    print("Cargando modelo base...")
    modelo_base = keras.models.load_model(BASE_MODEL_PATH)

    modelo_podado = aplicar_pruning(modelo_base, x_train, y_train, x_test, y_test)
    convertir_a_tflite(modelo_podado, x_test[:100])

    print("\nListo. Ejecuta 'python src/benchmark.py' para comparar los tres artefactos.")


if __name__ == "__main__":
    main()
