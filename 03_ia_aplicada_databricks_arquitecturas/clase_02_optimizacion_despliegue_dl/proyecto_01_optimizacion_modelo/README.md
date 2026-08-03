# Proyecto 1 — Optimización de un modelo de Deep Learning

Entrena una CNN pequeña sobre **Fashion-MNIST**, la optimiza con **pruning** + **quantization**, y compara el resultado contra el modelo original. Es el primer eslabón del pipeline de la clase: el `.tflite` que produce este proyecto es el que consume `proyecto_02_api_fastapi_docker/`.

```text
proyecto_01_optimizacion_modelo/
├── README.md
├── requirements.txt
├── src/
│   ├── train_model.py       ← entrena y guarda models/modelo_base.h5
│   ├── optimize_model.py    ← pruning + quantization -> models/modelo_optimizado.tflite
│   └── benchmark.py         ← compara tamaño/latencia/accuracy
└── models/                  ← artefactos generados (no versionados salvo .gitkeep)
```

## Requisitos

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución paso a paso

```bash
# 1. Entrenar el modelo base (≈2-4 min en CPU)
python src/train_model.py

# 2. Aplicar pruning + fine-tuning + quantization a TFLite
python src/optimize_model.py

# 3. Comparar tamaño, latencia y precisión
python src/benchmark.py
```

Al finalizar tendrás en `models/`:

| Archivo | Descripción |
|---|---|
| `modelo_base.h5` | CNN original, sin optimizar |
| `modelo_podado.h5` | CNN con pruning aplicado (sin cuantizar) |
| `modelo_optimizado.tflite` | Modelo podado + cuantizado, listo para servir |
| `comparacion.md` | Tabla con los resultados del benchmark |

> **Nota de compatibilidad:** desde TensorFlow 2.16, Keras 3 es el backend por defecto, pero `tensorflow-model-optimization` (usado para el pruning) todavía no lo soporta. Por eso `train_model.py` y `optimize_model.py` fijan `TF_USE_LEGACY_KERAS=1` internamente y `requirements.txt` incluye `tf-keras`. No necesitas hacer nada extra: basta con `pip install -r requirements.txt`.

## Qué hace cada script

**`train_model.py`** — carga Fashion-MNIST, construye una CNN de 2 bloques convolucionales + capa densa, entrena 5 épocas y guarda `modelo_base.h5`.

**`optimize_model.py`** — aplica poda gradual de magnitud (`tensorflow-model-optimization`, `final_sparsity=0.5`) con 2 épocas de fine-tuning para recuperar precisión, y luego convierte el modelo a TFLite con `tf.lite.Optimize.DEFAULT` (quantization dinámica).

**`benchmark.py`** — carga ambos modelos, mide accuracy sobre el set de test completo, mide latencia promedio de inferencia individual sobre 200 muestras, y compara tamaño en disco. Genera `models/comparacion.md`.

## Resultado esperado (orden de magnitud)

| Métrica | Base (.h5) | Optimizado (.tflite) |
|---|---|---|
| Tamaño | ~0.6–1 MB | ~0.15–0.3 MB |
| Accuracy | ~90–91% | ~88–90% (pérdida menor a 2 pts) |
| Latencia | mayor | menor |

Los valores exactos dependen del hardware donde se ejecute. Lo importante es la **dirección** del cambio: modelo más chico y más rápido, con una pérdida de precisión pequeña y aceptable.

## Siguiente paso

Con `models/modelo_optimizado.tflite` generado, continúa en `../proyecto_02_api_fastapi_docker/` para exponerlo como servicio REST.
