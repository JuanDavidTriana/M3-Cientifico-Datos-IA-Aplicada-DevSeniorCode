"""API que sirve el modelo BASE (sin optimizar). Ver servicio_optimizado/app/main.py
para la contraparte con el modelo podado + cuantizado."""
import logging

import numpy as np
from fastapi import FastAPI, HTTPException

from app.model_utils import obtener_modelo
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baseline-api")

app = FastAPI(title="Servicio de inferencia — modelo BASE (sin optimizar)", version="1.0.0")

_modelo = None


@app.on_event("startup")
def cargar_modelo_al_iniciar():
    global _modelo
    try:
        _modelo = obtener_modelo()
        logger.info("Modelo base cargado desde %s", _modelo.model_path)
    except FileNotFoundError as exc:
        logger.warning("El modelo no está disponible todavía: %s", exc)
        _modelo = None


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        modelo_cargado=_modelo is not None,
        version_modelo="modelo_base.h5" if _modelo else "no_cargado",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if _modelo is None:
        raise HTTPException(status_code=503, detail="El modelo base no está cargado.")
    try:
        imagen = np.array(request.imagen)
        resultado = _modelo.predecir(imagen)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PredictionResponse(**resultado)
