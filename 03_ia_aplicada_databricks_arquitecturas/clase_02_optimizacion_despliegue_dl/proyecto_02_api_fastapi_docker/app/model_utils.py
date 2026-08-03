"""
Utilidades para cargar el modelo TFLite optimizado y ejecutar inferencia.

El modelo se carga UNA sola vez (patrón singleton) cuando el módulo se importa,
para que main.py nunca vuelva a leer el archivo .tflite en cada petición.
"""
import os
import time

import numpy as np
import tensorflow as tf

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "model", "modelo_optimizado.tflite"),
)

CLASS_NAMES = [
    "Camiseta/top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín",
]


class ModeloOptimizado:
    """Envuelve un intérprete de TFLite para hacer inferencia sencilla."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo en '{model_path}'. "
                "Copia aquí el 'modelo_optimizado.tflite' generado en "
                "proyecto_01_optimizacion_modelo/models/."
            )
        self.interprete = tf.lite.Interpreter(model_path=model_path)
        self.interprete.allocate_tensors()
        self.entrada = self.interprete.get_input_details()[0]
        self.salida = self.interprete.get_output_details()[0]
        self.model_path = model_path

    def predecir(self, imagen: np.ndarray) -> dict:
        """Recibe una imagen (28, 28) y devuelve clase, confianza y latencia."""
        imagen = self._preprocesar(imagen)

        inicio = time.perf_counter()
        self.interprete.set_tensor(self.entrada["index"], imagen)
        self.interprete.invoke()
        salida = self.interprete.get_tensor(self.salida["index"])[0]
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

        # Si los valores parecen estar en [0, 255], normalizar a [0, 1]
        if imagen.max() > 1.0:
            imagen = imagen / 255.0

        imagen = imagen.reshape(1, 28, 28, 1).astype(self.entrada["dtype"])
        return imagen


_modelo_singleton: ModeloOptimizado | None = None


def obtener_modelo() -> ModeloOptimizado:
    global _modelo_singleton
    if _modelo_singleton is None:
        _modelo_singleton = ModeloOptimizado(MODEL_PATH)
    return _modelo_singleton
