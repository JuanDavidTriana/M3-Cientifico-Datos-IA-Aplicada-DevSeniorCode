# Proyecto 2 — API con FastAPI + Docker

Expone el modelo optimizado (`modelo_optimizado.tflite`, generado en `proyecto_01_optimizacion_modelo/`) como un servicio REST, empaquetado en una imagen Docker lista para desplegar.

```text
proyecto_02_api_fastapi_docker/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── test_api.py            ← cliente de prueba simple
├── app/
│   ├── main.py             ← endpoints /health y /predict
│   ├── model_utils.py      ← carga el modelo (una sola vez) e infiere
│   └── schemas.py          ← validación de entrada/salida (Pydantic)
└── model/                  ← aquí va modelo_optimizado.tflite (no versionado)
```

## Paso 0 — copiar el modelo optimizado

Este proyecto **no entrena nada**; consume el artefacto del proyecto 1.

```bash
# Desde clase_02_optimizacion_despliegue_dl/
cp proyecto_01_optimizacion_modelo/models/modelo_optimizado.tflite \
   proyecto_02_api_fastapi_docker/model/modelo_optimizado.tflite
```

En Windows (PowerShell):

```powershell
Copy-Item proyecto_01_optimizacion_modelo\models\modelo_optimizado.tflite `
          proyecto_02_api_fastapi_docker\model\modelo_optimizado.tflite
```

## Opción A — ejecutar localmente (sin Docker)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abre `http://localhost:8000/docs` para la documentación interactiva, o prueba con:

```bash
python test_api.py
```

## Opción B — ejecutar con Docker

```bash
# Construir la imagen
docker build -t clase02-inference-api .

# Levantar el contenedor
docker run --rm -p 8000:8000 clase02-inference-api
```

Con el contenedor corriendo, en otra terminal:

```bash
python test_api.py --host http://localhost:8000
```

O directamente con `curl`:

```bash
curl http://localhost:8000/health
```

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servicio y si el modelo está cargado |
| `POST` | `/predict` | Recibe `{"imagen": [[...28 valores...], ... 28 filas]}` y devuelve clase, confianza y latencia |
| `GET` | `/docs` | Documentación interactiva (Swagger UI) |

Ejemplo de petición a `/predict`:

```json
{
  "imagen": [[0, 0, ..., 0], [0, 12, 200, ..., 0], "... 28 filas de 28 valores ..."]
}
```

Respuesta:

```json
{
  "clase_id": 7,
  "clase_nombre": "Zapatilla",
  "confianza": 0.93,
  "latencia_ms": 1.2
}
```

## Configuración con `.env`

`app/model_utils.py` lee la ruta del modelo desde la variable de entorno `MODEL_PATH`, con un valor por defecto si no se define. Esto permite cambiar dónde vive el modelo sin tocar el código, algo esencial al pasar de tu máquina a un servidor real.

```bash
cp .env.example .env
# edita .env si necesitas otra ruta u otro nivel de log

docker run --env-file .env -p 8000:8000 clase02-inference-api
```

El `.env` real nunca se versiona (ver `.gitignore` en la raíz del repo); solo se versiona `.env.example`. Más contexto en `../README.md`, sección 9.

## Desplegar en un servidor remoto por SSH

Una vez la imagen funciona localmente, llevarla a un servidor real es, en esencia, tres comandos:

```bash
# 1. Copiar el código y el modelo al servidor
scp -r . usuario@servidor.ejemplo.com:/opt/clase02-api/

# 2. Conectarte por SSH
ssh usuario@servidor.ejemplo.com

# 3. En el servidor: construir y levantar el contenedor
cd /opt/clase02-api && docker build -t clase02-inference-api . && \
  docker run -d --env-file .env -p 8000:8000 clase02-inference-api
```

Ver la explicación completa de SSH (llaves, buenas prácticas) en `../README.md`, sección 9.1.

## Decisiones de diseño

- **El modelo se carga una sola vez**, en el evento `startup` de FastAPI (`app/main.py`), nunca dentro del endpoint `/predict`. Cargar un modelo en cada petición sería el error de rendimiento más común al servir modelos.
- **`model_utils.py` está separado de `main.py`**: la lógica de inferencia es independiente del framework web, lo que facilita testearla o reutilizarla (por ejemplo, en `proyecto_03_benchmark_integrador/`).
- **El Dockerfile copia `requirements.txt` antes que el código** para aprovechar el caché de capas: reconstruir la imagen tras un cambio de código no reinstala dependencias.
- **`HEALTHCHECK`** en el Dockerfile permite que Docker (o un orquestador) detecte si el contenedor sigue funcionando correctamente.

## Siguiente paso

Continúa en `../proyecto_03_benchmark_integrador/` para comparar este servicio optimizado contra un servicio equivalente con el modelo base, usando `docker-compose`.
