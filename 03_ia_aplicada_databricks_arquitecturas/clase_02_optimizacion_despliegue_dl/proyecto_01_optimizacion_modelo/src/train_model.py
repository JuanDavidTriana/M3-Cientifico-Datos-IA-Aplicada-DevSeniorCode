"""
Entrena una CNN pequeña sobre Fashion-MNIST y la guarda como modelo base (.h5).

Este es el modelo "de partida" que en optimize_model.py se poda y se cuantiza
para producir una versión liviana lista para despliegue.

Uso:
    python src/train_model.py
"""
import os

# tensorflow-model-optimization (usado en optimize_model.py) todavía no es
# compatible con Keras 3, el backend por defecto desde TensorFlow 2.16.
# Fijamos Keras legacy (tf_keras) desde el entrenamiento para que el .h5
# resultante sea compatible con todo el pipeline de optimización.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow import keras

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BASE_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_base.h5")

CLASS_NAMES = [
    "Camiseta/top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín",
]


def cargar_datos():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalizar a [0, 1] y agregar canal (28, 28) -> (28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    return (x_train, y_train), (x_test, y_test)


def construir_modelo():
    modelo = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ], name="cnn_fashion_mnist")

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Cargando Fashion-MNIST...")
    (x_train, y_train), (x_test, y_test) = cargar_datos()

    print("Construyendo modelo...")
    modelo = construir_modelo()
    modelo.summary()

    print("Entrenando modelo base (5 épocas)...")
    modelo.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=128,
        verbose=2,
    )

    test_loss, test_acc = modelo.evaluate(x_test, y_test, verbose=0)
    print(f"\nAccuracy en test (modelo base): {test_acc:.4f}")

    modelo.save(BASE_MODEL_PATH)
    print(f"Modelo base guardado en: {BASE_MODEL_PATH}")


if __name__ == "__main__":
    main()
