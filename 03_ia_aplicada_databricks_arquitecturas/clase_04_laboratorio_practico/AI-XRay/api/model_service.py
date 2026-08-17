"""
Carga del modelo entrenado y lógica de inferencia, separada de los endpoints de FastAPI
(`main.py`) -- así el modelo se carga UNA sola vez al iniciar el proceso, no en cada
petición (una de las buenas prácticas ya vistas en la Clase 2 para servir modelos).

Dos formas de obtener el modelo (controladas por la variable de entorno
`AIXRAY_MODEL_SOURCE`, por defecto "local"):

- "local": carga un archivo `.keras` ya exportado (lo que genera el notebook
  04_evaluation.ipynb al final, sobre `models/`). No depende de tener un servidor de
  MLflow corriendo -- es la forma más simple de "levantar el backend y probarlo".
- "mlflow": carga el modelo directamente desde el Model Registry de MLflow
  (`models:/aixray_pneumonia_classifier/Staging`), útil cuando sí se quiere demostrar
  el ciclo completo de MLOps (registro -> promoción -> consumo) en vivo.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model as keras_load_model

from src.config import CLASSES, IMG_SIZE, MLFLOW_MODEL_NAME, MODELS_DIR

MODEL_SOURCE = "local"
LOCAL_MODEL_PATH = MODELS_DIR / "aixray_model.keras"
MLFLOW_MODEL_STAGE = "Staging"

MODEL_VERSION_LABEL = f"local:{LOCAL_MODEL_PATH.name}" if MODEL_SOURCE == "local" else f"mlflow:{MLFLOW_MODEL_STAGE}"


class ModelService:
    """Envoltorio simple: carga el modelo una vez y expone `predict_image(bytes)`."""

    def __init__(self):
        self.model = None
        self.version_label = MODEL_VERSION_LABEL

    def load(self):
        if self.model is not None:
            return
        if MODEL_SOURCE == "local":
            if not LOCAL_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"No se encontró el modelo en {LOCAL_MODEL_PATH}. Corre primero "
                    "notebooks/03_training.ipynb y 04_evaluation.ipynb para entrenar, "
                    "evaluar y exportar el modelo, o define AIXRAY_MODEL_PATH."
                )
            self.model = keras_load_model(LOCAL_MODEL_PATH)
        else:
            import mlflow.keras

            model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
            self.model = mlflow.keras.load_model(model_uri)

    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.asarray(img, dtype="float32")
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def predict_image(self, image_bytes: bytes) -> dict:
        if self.model is None:
            self.load()
        x = self.preprocess(image_bytes)
        prob_pneumonia = float(self.model.predict(x, verbose=0)[0][0])
        label = CLASSES[1] if prob_pneumonia >= 0.5 else CLASSES[0]
        return {
            "prediction": label,
            "probability": prob_pneumonia,
            "model_version": self.version_label,
        }


# Instancia única compartida por toda la app (patrón singleton simple).
model_service = ModelService()
