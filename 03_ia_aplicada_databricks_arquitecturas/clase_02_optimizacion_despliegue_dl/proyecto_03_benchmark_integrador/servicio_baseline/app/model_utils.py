"""Carga el modelo BASE (.h5, sin optimizar) y ejecuta inferencia con Keras.

Sirve como contraparte "sin optimizar" del servicio en proyecto_02, para que
benchmark/compare.py pueda medir la diferencia real de latencia entre ambos.
"""
import os

# El .h5 fue guardado con Keras legacy (ver nota en proyecto_01/src/train_model.py);
# hay que fijar esto ANTES de importar tensorflow para poder cargarlo.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import time

import numpy as np
import tensorflow as tf

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "model", "modelo_base.h5"),
)

CLASS_NAMES = [
    "Camiseta/top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín",
]


class ModeloBase:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo en '{model_path}'. "
                "Copia aquí el 'modelo_base.h5' generado en "
                "proyecto_01_optimizacion_modelo/models/."
            )
        self.modelo = tf.keras.models.load_model(model_path)
        self.model_path = model_path

    def predecir(self, imagen: np.ndarray) -> dict:
        imagen = self._preprocesar(imagen)

        inicio = time.perf_counter()
        salida = self.modelo.predict(imagen, verbose=0)[0]
        latencia_ms = (time.perf_counter() - inicio) * 1000

        clase_id = int(np.argmax(salida))
        confianza = float(salida[clase_id])

        return {
            "clase_id": clase_id,
            "clase_nombre": CLASS_NAMES[clase_id],
            "confianza": confianza,
            "latencia_ms": round(latencia_ms, 3),
        }

    def _preprocesar(self, imagen: np.ndarray) -> np.ndarray:
        imagen = np.asarray(imagen, dtype="float32")
        if imagen.shape != (28, 28):
            raise ValueError(f"Se esperaba una imagen de 28x28, se recibió {imagen.shape}")
        if imagen.max() > 1.0:
            imagen = imagen / 255.0
        return imagen.reshape(1, 28, 28, 1)


_modelo_singleton: "ModeloBase | None" = None


def obtener_modelo() -> ModeloBase:
    global _modelo_singleton
    if _modelo_singleton is None:
        _modelo_singleton = ModeloBase(MODEL_PATH)
    return _modelo_singleton
