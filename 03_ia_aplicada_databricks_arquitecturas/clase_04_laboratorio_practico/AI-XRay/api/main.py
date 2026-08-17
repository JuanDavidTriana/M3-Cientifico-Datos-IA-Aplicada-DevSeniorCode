"""
AI-XRay -- API de inferencia (FastAPI).

Este es TODO el "producto" de la Fase 1: un backend que carga el modelo ya entrenado
y expone un endpoint para clasificar una radiografía. No hay frontend en este proyecto
-- la forma de probarlo es la documentación interactiva automática en `/docs` (Swagger)
o un cliente simple (ver `tests/test_api.py` o un `curl`, documentado en el README).

Levantar en local:
    uvicorn api.main:app --reload --port 8000
Luego abrir: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from api.model_service import model_service
from api.schemas import ErrorResponse, HealthResponse, PredictionResponse

DISCLAIMER = (
    "Este sistema es un proyecto académico de inteligencia artificial y no constituye "
    "un diagnóstico médico ni sustituye la evaluación de un profesional de la salud."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar el modelo UNA sola vez al iniciar el proceso, no en cada petición.
    try:
        model_service.load()
    except FileNotFoundError as e:
        # No tumbamos el proceso: /health reportará model_loaded=False y /predict
        # devolverá un 503 claro, en vez de que la app ni siquiera arranque.
        print(f"[startup] Aviso: {e}")
    yield


app = FastAPI(
    title="AI-XRay -- API de inferencia",
    description=(
        "Clasificación de radiografías de tórax pediátricas (NORMAL / PNEUMONIA) con "
        "ResNet50 + Transfer Learning. Proyecto académico. " + DISCLAIMER
    ),
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}


@app.get("/health", response_model=HealthResponse, tags=["monitoreo"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded(),
        model_version=model_service.version_label,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
    tags=["inferencia"],
)
async def predict(file: UploadFile = File(..., description="Imagen de radiografía (JPEG/PNG)")):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {file.content_type}. Usa JPEG o PNG.",
        )

    if not model_service.is_loaded():
        try:
            model_service.load()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

    image_bytes = await file.read()
    try:
        result = model_service.predict_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo procesar la imagen: {e}")

    return {**result, "disclaimer": DISCLAIMER}
