"""
Pruebas mínimas de la API. Requieren que exista un modelo exportado en
`models/aixray_model.keras` (lo genera `notebooks/04_evaluation.ipynb`) o que se
apunte `AIXRAY_MODEL_PATH` a uno de prueba.

Correr con:  pytest tests/ -v
"""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)


def _fake_jpeg_bytes() -> bytes:
    """Genera una imagen JPEG sintética en memoria (no una radiografía real) solo para
    probar que el endpoint procesa correctamente el formato de entrada."""
    img = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_rejects_wrong_content_type():
    response = client.post(
        "/predict", files={"file": ("doc.txt", b"no soy una imagen", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_returns_valid_prediction_shape():
    image_bytes = _fake_jpeg_bytes()
    response = client.post("/predict", files={"file": ("xray.jpg", image_bytes, "image/jpeg")})
    if response.status_code == 503:
        # No hay modelo exportado en este entorno de pruebas: se documenta el caso en
        # vez de fallar el test -- ver docstring del módulo.
        assert "detail" in response.json()
        return
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in ("NORMAL", "PNEUMONIA")
    assert 0.0 <= body["probability"] <= 1.0
    assert "disclaimer" in body
