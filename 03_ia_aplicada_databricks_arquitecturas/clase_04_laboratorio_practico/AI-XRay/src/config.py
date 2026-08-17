"""
Configuración central del proyecto AI-XRay.

Todas las rutas y constantes que otros módulos (notebooks, src/, api/) necesitan
viven aquí para no repetirlas ni desincronizarlas entre archivos.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------

# Raíz del proyecto AI-XRay/ (donde vive este archivo, dos niveles arriba: src/config.py -> AI-XRay/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# El dataset "Pediatric Chest X-ray Pneumonia" (Kaggle, andrewmvd) ya está descargado
# y descomprimido un nivel arriba de AI-XRay/, dentro de la carpeta de la Clase 4:
#   clase_04_laboratorio_practico/
#   ├── archive (2)/Pediatric Chest X-ray Pneumonia/{train,test}/{NORMAL,PNEUMONIA}
#   └── AI-XRay/                                          <- PROJECT_ROOT
#
# Se puede sobreescribir con la variable de entorno AIXRAY_DATA_DIR sin tocar código,
# por ejemplo si mueves las imágenes a otra carpeta (ver data/README.md).
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "archive (2)" / "Pediatric Chest X-ray Pneumonia"
DATA_DIR = Path(os.environ.get("AIXRAY_DATA_DIR", str(DEFAULT_DATA_DIR)))

# Carpetas de salida: se crean solas la primera vez que un notebook escribe en ellas
# (mkdir(parents=True, exist_ok=True)) -- no hace falta que existan de antemano en el
# repositorio ni crearlas a mano.
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest_split.csv"

# Tracking de MLflow: se usa un backend SQLite local. Databricks Free Edition
# no necesita esto (ya trae tracking gestionado); fuera de Databricks, las versiones
# recientes de MLflow (>=3.x) ya no soportan el filestore clásico "./mlruns" para
# nuevas instalaciones, así que usamos SQLite desde el día uno.
MLFLOW_TRACKING_URI = os.environ.get(
    "AIXRAY_MLFLOW_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
MLFLOW_EXPERIMENT_NAME = "AI-XRay"
MLFLOW_MODEL_NAME = "aixray_pneumonia_classifier"

# --------------------------------------------------------------------------
# Constantes del problema
# --------------------------------------------------------------------------

CLASSES = ["NORMAL", "PNEUMONIA"]  # índice 0 = NORMAL, índice 1 = PNEUMONIA (v1 binaria)
IMG_SIZE = (224, 224)  # tamaño esperado por ResNet50 con pesos de ImageNet
IMG_CHANNELS = 3
BATCH_SIZE = 32

# Subconjunto usado para la demo/laboratorio de clase (dataset completo = 5,856 imágenes).
# Balanceado a propósito para que el entrenamiento en vivo no tenga que lidiar con
# desbalance de clases -- ver README.md, sección "Por qué un subconjunto balanceado".
# Configurable con AIXRAY_SUBSET_TOTAL (útil, por ejemplo, para correr el notebook con
# una muestra pequeña al probar el pipeline por primera vez).
DEMO_SUBSET_TOTAL = int(os.environ.get("AIXRAY_SUBSET_TOTAL", 2000))
DEMO_SUBSET_PER_CLASS = DEMO_SUBSET_TOTAL // len(CLASSES)  # 1000 NORMAL + 1000 PNEUMONIA

# Split propio (NO se usa el split original train/test del dataset tal cual, ver README).
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

SEED = 42
