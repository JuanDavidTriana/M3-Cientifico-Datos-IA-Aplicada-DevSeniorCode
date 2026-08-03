"""
API REST que sirve el modelo optimizado (.tflite) entrenado en el proyecto 1.

Endpoints:
    GET  /health   -> estado del servicio
    POST /predict  -> clasifica una imagen 28x28 (Fashion-MNIST)

Ejecución local:
    uvicorn app.main:app --reload

Documentación interactiva una vez levantado el servicio: http://localhost:8000/docs
"""
import logging

import numpy as np
from fastapi import FastAPI, HTTPException

from app.model_utils import obtener_modelo
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference-api")

app = FastAPI(
    title="Servicio de inferencia — Fashion-MNIST optimizado",
    description="Sirve un modelo CNN podado y cuantizado (TFLite) entrenado en el proyecto 1 de la clase.",
    version="1.0.0",
)

_modelo = None


@app.on_event("startup")
def cargar_modelo_al_iniciar():
    """Carga el modelo una sola vez, al arrancar la aplicación (no en cada petición)."""
    global _modelo
    try:
        _modelo = obtener_modelo()
        logger.info("Modelo cargado correctamente desde %s", _modelo.model_path)
    except FileNotFoundError as exc:
        logger.warning("El modelo no está disponible todavía: %s", exc)
        _modelo = None


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        modelo_cargado=_modelo is not None,
        version_modelo="modelo_optimizado.tflite" if _modelo else "no_cargado",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if _modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está cargado. Verifica que 'model/modelo_optimizado.tflite' exista.",
        )

    try:
        imagen = np.array(request.imagen)
        resultado = _modelo.predecir(imagen)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictionResponse(**resultado)
