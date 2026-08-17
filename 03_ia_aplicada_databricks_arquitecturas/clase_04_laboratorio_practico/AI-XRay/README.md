# AI-XRay

Clasificación de radiografías de tórax pediátricas (NORMAL / PNEUMONIA) con
**CNN + Transfer Learning (ResNet50)**, entrenada con **TensorFlow/Keras**, con
seguimiento completo de experimentos en **MLflow** y servida mediante una **API REST
(FastAPI)**.

Proyecto de laboratorio de la Clase 4 (Unidad 3, Módulo de Ciencia de Datos e IA
Aplicada): el docente construye este sistema en vivo, de punta a punta, y cada
estudiante ejecuta después su propia instancia independiente.

> **Aviso:** este sistema es un proyecto académico de inteligencia artificial y **no
> constituye un diagnóstico médico** ni sustituye la evaluación de un profesional de
> la salud. Este mismo aviso se muestra en cada respuesta de la API (campo
> `disclaimer`) y debe repetirse en el Executive Summary de cada estudiante.

---

## 1. El flujo de punta a punta

```text
Dataset (Kaggle)
   ↓
Manifiesto + EDA (pandas / PySpark simbólico)
   ↓
Split propio 70/15/15, agrupado por paciente
   ↓
Preprocesamiento + Data Augmentation (tf.data)
   ↓
CNN + Transfer Learning (ResNet50)
   ↓
Entrenamiento (Fase A: congelado · Fase B: fine-tuning)
   ↓
Evaluación (Accuracy, Precision, Recall, F1, ROC-AUC, matriz de confusión)
   ↓
MLflow (params, métricas, artefactos, modelo, Model Registry)
   ↓
Modelo exportado (models/aixray_model.keras)
   ↓
API de inferencia (FastAPI: POST /predict, GET /health)
```

**No hay frontend en este proyecto.** El entregable de "aplicación" es únicamente el
backend: se prueba con la documentación interactiva de FastAPI (`/docs`) o con un
cliente simple (`curl`, o `tests/test_api.py`). Ver la sección 7.

---

## 2. Dataset

Ver [`data/README.md`](data/README.md) para el detalle completo: fuente, cita
obligatoria, dónde debe colocarse, composición real confirmada (5,856 imágenes:
1,583 NORMAL / 4,273 PNEUMONIA), por qué este proyecto **no** usa el split
`train/`/`test/` original (fuga de datos a nivel paciente, confirmada con un ejemplo
real del propio dataset), y el subconjunto de 2,000 imágenes balanceadas usado en la
demo de clase.

El diseño del código (`label` numérico + `label_name` string en el manifiesto, y
`num_classes` como parámetro en `build_model`) permite ampliar a clasificación
multiclase más adelante (por ejemplo, separar `PNEUMONIA` en bacteriana/viral, una
distinción que el propio nombre de archivo del dataset ya trae) sin rediseñar el
pipeline.

---

## 3. Arquitectura del modelo

```text
Imagen (JPEG)
   ↓ resize a 224×224
   ↓ preprocess_input (estadísticas de ImageNet — NO normalización genérica 0-1)
   ↓ data augmentation (solo en train: flip, rotación, zoom, contraste)
ResNet50 (pesos de ImageNet, sin la capa top)
   ↓
GlobalAveragePooling2D
   ↓
Dropout
   ↓
Dense(1, activation="sigmoid")
   ↓
NORMAL (0) / PNEUMONIA (1)
```

Entrenamiento en dos fases (ambas implementadas en `src/architecture.py` y
orquestadas desde `notebooks/03_training.ipynb`):

- **Fase A -- clasificador congelado:** toda la base de ResNet50 permanece congelada
  (`base_trainable=False`); solo se entrena el clasificador nuevo (GAP + Dropout +
  Dense). Rápido, y el punto de partida obligatorio antes de cualquier fine-tuning.
- **Fase B -- fine-tuning:** se descongelan las últimas capas de la base
  (`fine_tune_at`, por defecto la capa 143 de 175) con un *learning rate* mucho más
  bajo (`1e-5` en el run de referencia), para adaptar los filtros más específicos de
  ResNet50 al dominio de rayos X sin destruir los features genéricos aprendidos en
  ImageNet.

---

## 4. Entrenamiento

Implementado en `src/train.py` (`train_and_log`), usado por
`notebooks/03_training.ipynb`:

- Split propio train/val/test (70/15/15), agrupado por paciente.
- Data augmentation (solo en train).
- `EarlyStopping` (monitorea `val_loss`, `patience=4`, restaura los mejores pesos).
- `ReduceLROnPlateau` (reduce el *learning rate* a la mitad si `val_loss` se estanca).
- `ModelCheckpoint` (guarda el mejor modelo de cada run).
- `class_weight` calculado con `compute_balanced_class_weight()` -- en el subconjunto
  balanceado de la demo da pesos ~1.0/1.0 (no hace nada); su utilidad real aparece si
  se entrena con el dataset completo desbalanceado.
- Semillas fijas (`SEED=42` en `src/config.py`) vía `set_seeds()`.
- El modelo final de cada run se guarda automáticamente en MLflow; el mejor run se
  exporta además como archivo local (`models/aixray_model.keras`) para que la API lo
  cargue sin depender de un servidor de MLflow corriendo.

### Runs de referencia (los mismos 3 que arma el docente en vivo)

| Run | Arquitectura | Learning rate | Dropout |
|---|---|---|---|
| 1 | ResNet50 congelado | 0.001 | 0.3 |
| 2 | ResNet50 congelado | 0.0001 | 0.5 |
| 3 | ResNet50 + Fine-tuning | 0.00001 | 0.3 |

Cada estudiante debe correr **al menos 2 variantes propias** (puede reutilizar estas 3
como punto de partida y ajustar hiperparámetros) y compararlas en MLflow antes de
elegir cuál registrar.

---

## 5. Evaluación

Implementado en `src/metrics.py`. Nunca solo *accuracy* -- en un problema
médico con clases desbalanceadas, un modelo que siempre predice "PNEUMONIA" tendría
accuracy alta y sería clínicamente inútil. Se generan y registran en MLflow:

- Matriz de confusión (imagen)
- Curva ROC (imagen)
- Curvas de *training loss/accuracy* vs. *validation loss/accuracy* (imagen)
- Classification report completo (texto): precision, recall, F1 por clase
- Accuracy, Precision, Recall, F1-score, ROC-AUC (métricas numéricas en MLflow)

---

## 6. MLflow

`Experiment: AI-XRay` (ver `src/config.py`). Cada run registra:

- **Parámetros:** arquitectura, learning rate, dropout, épocas, batch size,
  optimizador, fine-tuning (sí/no y en qué capa), semilla, `class_weight`.
- **Métricas:** loss/accuracy por época (train y val) + accuracy, precision, recall,
  F1, AUC finales.
- **Artefactos:** matriz de confusión, curva ROC, curvas de entrenamiento,
  classification report.
- **El modelo**, con *model signature* (`mlflow.models.infer_signature`).

> **Nota técnica (workaround verificado):** en la combinación de versiones fijada en
> `requirements.txt` (MLflow 3.15.1 + Keras 3), `mlflow.keras.log_model(...,
> input_example=...)` falla con un `FileNotFoundError` interno al serializar el
> ejemplo de entrada -- es un problema conocido de esa combinación de librerías, no del
> código de este proyecto. El workaround (ya aplicado en `train_and_log`): se omite
> `input_example` en `log_model` y se registra la forma de entrada esperada como un
> artefacto JSON aparte (`input_example_info.json`). El efecto práctico -- documentar
> qué entrada espera el modelo -- es el mismo.

Una vez comparados los runs (`notebooks/04_evaluation.ipynb`), el mejor se registra en
el **Model Registry** (`mlflow.register_model`) y se promueve a `Staging`
(`MlflowClient().transition_model_version_stage`), siguiendo el mismo patrón visto en
la Clase 3.

---

## 7. Backend / API (FastAPI)

Único entregable de "aplicación" de este proyecto -- sin frontend, sin contenedores:
un proceso de Python que se levanta directamente con `uvicorn`.

```bash
uvicorn api.main:app --reload --port 8000
```

Luego abrir `http://localhost:8000/docs` (Swagger, generado automáticamente) para
probar `/predict` subiendo una imagen desde el navegador, sin escribir código.

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | `{"status": "ok", "model_loaded": true, "model_version": "..."}` |
| `/predict` | POST | Recibe un archivo de imagen (`multipart/form-data`), devuelve `{"prediction": "PNEUMONIA", "probability": 0.942, "model_version": "1.0", "disclaimer": "..."}` |
| `/docs` | GET | Documentación interactiva (Swagger UI) |

Ejemplo con `curl`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@ruta/a/una/radiografia.jpeg"
```

El modelo se carga **una sola vez** al iniciar el proceso (`api/model_service.py`), no
en cada petición. Por defecto carga el archivo local `models/aixray_model.keras`
(`AIXRAY_MODEL_SOURCE=local`); puede configurarse para cargar en cambio desde el
Model Registry de MLflow (`AIXRAY_MODEL_SOURCE=mlflow`), útil si se quiere demostrar
el ciclo completo de MLOps en vivo.

Pruebas automatizadas: `pytest tests/ -v` (verificadas: `/health`, rechazo de
archivos que no son imagen, forma válida de la respuesta de `/predict`).

---

## 8. Estructura del proyecto

Estructura deliberadamente plana: cada carpeta existe porque contiene algo que se usa,
no como andamiaje "por si acaso". `models/` y `mlruns/` no aparecen aquí porque no se
suben al repositorio -- los crea Python solo la primera vez que corres los notebooks.

```text
AI-XRay/
├── data/
│   └── README.md          fuente del dataset, cita obligatoria, dónde colocarlo
├── notebooks/
│   ├── 01_exploracion.ipynb       manifiesto, EDA, split, chequeo de fuga de datos
│   ├── 02_preprocessing.ipynb     pipeline tf.data, augmentation, verificación visual
│   ├── 03_training.ipynb          3 runs de MLflow (Fase A x2 + Fase B fine-tuning)
│   └── 04_evaluation.ipynb        comparación de runs, registro, promoción, export
├── src/
│   ├── config.py           rutas y constantes centralizadas (lo único que todos importan)
│   ├── dataset.py          manifiesto del dataset, chequeo de fuga, split por paciente
│   ├── preprocessing.py    pipeline tf.data: resize, preprocess_input, augmentation
│   ├── architecture.py     ResNet50 + Transfer Learning (build_model, compile_model)
│   ├── train.py            entrena una variante y registra todo en MLflow
│   └── metrics.py          matriz de confusión, ROC, curvas, classification report
├── api/
│   ├── main.py             endpoints FastAPI (POST /predict, GET /health)
│   ├── schemas.py          contratos Pydantic de request/response
│   └── model_service.py    carga el modelo una vez y expone predict_image()
├── tests/
│   └── test_api.py         pytest sobre la API (health, validación, forma de la respuesta)
├── requirements.txt
├── .gitignore
└── README.md               este archivo
```

**Por qué `src/` es plano y no tiene subcarpetas por tema:** con seis archivos en
total, una carpeta por archivo (`src/data/dataset.py`, `src/models/architecture.py`,
etc.) solo añade niveles que hay que abrir sin aportar nada -- cada módulo ya deja
claro qué hace en su docstring y en la tabla de arriba. Si el proyecto creciera mucho
más (por ejemplo, varias arquitecturas de modelo o varios pipelines de datos), ahí sí
tendría sentido volver a agrupar por tema.

---

## 9. Fase 1 (local) vs. Fase 2 (Databricks-ready)

Este proyecto se construye completo y funcional **en local** primero -- sin forzar
Databricks desde el día uno, que añadiría complejidad de infraestructura sin
beneficio real para ~2,000-6,000 imágenes. El único paso que ya se escribe "a la
manera de Databricks" es la construcción del manifiesto (`build_manifest_spark()` en
`src/dataset.py`, con PySpark), simbólico con este volumen de datos, pero
funcionalmente idéntico al que se usaría sobre un Data Lake real.

Este proyecto no implementa la Fase 2 -- queda documentada aquí, etapa por etapa, para
que cada estudiante entienda qué cambiaría si este mismo sistema se moviera a un
Workspace de Databricks (lo que se trabajó en la Clase 3: clusters, notebooks
colaborativos, MLflow gestionado):

| Etapa (versión local, esta Fase 1) | Equivalente en Databricks (Fase 2) | Qué cambia realmente |
|---|---|---|
| `src/dataset.py` — `build_manifest()` (pandas, recorrido local con `pathlib`) | `build_manifest_spark()` (ya incluido en el mismo archivo) leyendo desde un volumen/Unity Catalog o desde el Data Lake (ADLS/S3/GCS) montado en el Workspace | El código de Spark ya está escrito y probado en este proyecto; solo cambia la ruta de origen (de una carpeta local a un path de Data Lake) y el motor de cómputo (de un solo proceso a un cluster) |
| `data/manifest_split.csv` (archivo local) | Tabla Delta en Unity Catalog (`catalogo.esquema.aixray_manifest`) | El manifiesto se vuelve consultable con SQL por todo el equipo, versionado (time travel de Delta) y no vive en el disco de una sola persona |
| `notebooks/01_exploracion.ipynb` … `04_evaluation.ipynb` corridos localmente en Jupyter | Los mismos notebooks, subidos a un Workspace de Databricks y **adjuntados a un cluster** (o a Serverless) | El código de negocio (Spark, TensorFlow, MLflow) no cambia una línea; cambia dónde se ejecuta y quién tiene acceso colaborativo |
| `mlflow.set_tracking_uri("sqlite:///mlflow.db")` (backend local) | Tracking gestionado de Databricks (no requiere configurar nada: `mlflow.set_experiment("/Shared/aixray")` ya apunta al tracking server del Workspace) | Se elimina el archivo `mlflow.db` local; los experimentos quedan visibles para todo el equipo en la pestaña *Experiments* del Workspace |
| Entrenamiento de TensorFlow/Keras en un solo proceso (CPU o GPU local) | Entrenamiento distribuido con `spark-tensorflow-distributor` o `Petastorm` sobre un cluster con GPUs, o simplemente un cluster de un solo nodo con GPU si el dataset sigue siendo manejable | Solo se justifica si el dataset crece mucho más allá del subconjunto de 2,000 imágenes de este laboratorio (por ejemplo, si se usa el dataset completo NIH Chest X-ray14 con sus ~112,000 imágenes) |
| `models/aixray_model.keras` (archivo local cargado por `api/model_service.py`) | Modelo cargado desde el **Model Registry** de MLflow (`models:/aixray_pneumonia_classifier/Production`), tal como ya soporta `AIXRAY_MODEL_SOURCE=mlflow` en `model_service.py` | El código de la API ya está preparado para este modo -- ver `api/model_service.py`, no hace falta reescribirlo |
| `api/` corrida directamente con `uvicorn` en una máquina propia | La misma API, empaquetada en un contenedor (Docker) y desplegada en Databricks Model Serving, o en un servidor propio | El endpoint `/predict` no cambia de contrato; cambia de dónde obtiene el modelo y cómo se empaqueta para desplegarse de forma reproducible |

**Qué NO cambiaría:** el contrato de la API (`POST /predict`, `GET /health`) es
independiente de dónde se entrenó el modelo; la arquitectura (ResNet50 + Transfer
Learning) no cambia por el volumen de datos -- lo que cambia es cómo se *alimenta* el
entrenamiento; y el Executive Summary sigue siendo el mismo documento, solo que la
sección de "Arquitectura técnica" pasaría de describir un pipeline local a uno sobre
Databricks.

**Cuándo vale la pena migrar (y cuándo no):** migrar a Databricks tiene sentido cuando
el cuello de botella real es el **volumen de datos** (cientos de miles/millones de
imágenes) o la necesidad de **colaboración en equipo** sobre el mismo pipeline. Para un
laboratorio de clase con ~2,000-6,000 imágenes, forzar Databricks desde el día uno añade
complejidad de infraestructura sin un beneficio proporcional -- por eso esta Fase 1 se
construye completamente local, y esta Fase 2 queda documentada, no implementada.

---

## 10. Cómo correrlo de principio a fin

```bash
# 1. Entorno (requiere Python 3.10 o superior)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Notebooks, en orden
jupyter notebook notebooks/01_exploracion.ipynb   # manifiesto + EDA + split
jupyter notebook notebooks/02_preprocessing.ipynb # pipeline de datos + verificación visual
jupyter notebook notebooks/03_training.ipynb      # 3 runs de MLflow
jupyter notebook notebooks/04_evaluation.ipynb    # comparación, registro, export a models/

# 3. Revisar experimentos
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000

# 4. Levantar la API con el modelo ya exportado
uvicorn api.main:app --reload --port 8000           # http://localhost:8000/docs

# 5. Pruebas
pytest tests/ -v
```

---

## 11. Executive Summary

Cada estudiante entrega, además del código, un Executive Summary (plantilla en
`../executive_summary/plantilla_executive_summary.docx`, a nivel de la Clase 4) con:
Problema, Datos, Arquitectura, Modelo, Resultados, Experimentos, Mejor modelo,
Arquitectura de producción, Limitaciones, Impacto potencial -- incluyendo, en
Limitaciones, la nota sobre fuga de datos a nivel paciente descrita en
`data/README.md`, y el aviso de que este sistema no es una herramienta de diagnóstico
médico.

---

## Referencias

- Kermany D, Goldbaum M, Cai W, et al. *Identifying Medical Diagnoses and Treatable
  Diseases by Image-Based Deep Learning*. Cell. 2018;172(5):1122-1131.
- Dataset: [Pediatric Chest X-ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)
- [ResNet50 -- Keras Applications](https://keras.io/api/applications/resnet/)
- [MLflow -- Documentación oficial](https://mlflow.org/docs/latest/index.html)
- [FastAPI -- Documentación oficial](https://fastapi.tiangolo.com/)
