"""Esquemas de entrada/salida. Idénticos a los del servicio optimizado
para que benchmark/compare.py pueda hablarle a ambos servicios de la misma forma."""
from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    imagen: List[List[float]] = Field(..., min_length=28, max_length=28)


class PredictionResponse(BaseModel):
    clase_id: int
    clase_nombre: str
    confianza: float
    latencia_ms: float


class HealthResponse(BaseModel):
    status: str
    modelo_cargado: bool
    version_modelo: str
