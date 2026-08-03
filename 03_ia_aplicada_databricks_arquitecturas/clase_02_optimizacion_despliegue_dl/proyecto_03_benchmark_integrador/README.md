# Proyecto 3 — Benchmark integrador con Docker Compose

Levanta **dos servicios en paralelo** con `docker-compose` — uno con el modelo base y otro con el modelo optimizado del proyecto 1 — y los compara en vivo: latencia, throughput y tamaño del artefacto. Es el cierre del pipeline de la clase: aquí se cuantifica si la optimización realmente valió la pena.

```text
proyecto_03_benchmark_integrador/
├── README.md
├── docker-compose.yml
├── servicio_baseline/          ← API con el modelo SIN optimizar (puerto 8001)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   └── model/                  ← aquí va modelo_base.h5
├── servicio_optimizado/        ← API con el modelo optimizado (puerto 8002)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   └── model/                  ← aquí va modelo_optimizado.tflite
└── benchmark/
    ├── requirements.txt
    └── compare.py               ← mide y compara ambos servicios
```

## Paso 0 — copiar los modelos generados en el proyecto 1

```bash
# Desde clase_02_optimizacion_despliegue_dl/
cp proyecto_01_optimizacion_modelo/models/modelo_base.h5 \
   proyecto_03_benchmark_integrador/servicio_baseline/model/modelo_base.h5

cp proyecto_01_optimizacion_modelo/models/modelo_optimizado.tflite \
   proyecto_03_benchmark_integrador/servicio_optimizado/model/modelo_optimizado.tflite
```

## Paso 1 — levantar ambos servicios

```bash
cd proyecto_03_benchmark_integrador
docker compose up --build
```

Esto construye y levanta:

| Servicio | Puerto local | Modelo que sirve |
|---|---|---|
| `modelo-base` | `8001` | `modelo_base.h5` (sin optimizar) |
| `modelo-optimizado` | `8002` | `modelo_optimizado.tflite` (podado + cuantizado) |

Verifica que ambos estén saludables:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
```

## Paso 2 — ejecutar el benchmark comparativo

En otra terminal, con los contenedores corriendo:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r benchmark/requirements.txt
python benchmark/compare.py --n 100
```

El script envía el mismo lote de imágenes sintéticas a ambos servicios, mide latencia promedio, latencia p95 y throughput, y genera `benchmark/reporte.md`.

## Qué observar en los resultados

- **Latencia promedio y p95:** el modelo optimizado debería responder en menos tiempo por petición.
- **Throughput (req/s):** con el mismo tiempo total, el servicio optimizado debería atender más peticiones.
- **Tamaño del modelo:** impacta directamente el tiempo de arranque del contenedor y el costo de almacenamiento/transferencia de la imagen Docker.

Estos resultados son el mismo tipo de evidencia que se presenta en una decisión real de arquitectura: **¿vale la pena optimizar, dado el trade-off de precisión medido en el proyecto 1?**

## Apagar los servicios

```bash
docker compose down
```

## Configuración con `.env` (ejercicio 9 del README de la clase)

Este proyecto incluye `.env.example` como punto de partida. Para usarlo de verdad, copia el archivo (`cp .env.example .env`), agrega `env_file: .env` a cada servicio en `docker-compose.yml`, y ajusta `servicio_baseline/app/main.py` y `servicio_optimizado/app/main.py` para leer `LOG_LEVEL` desde el entorno en vez de tenerlo fijo en `logging.basicConfig(level=logging.INFO)`. Es exactamente el ejercicio 9 propuesto en `../README.md`.

## Ejercicio de cierre

Modifica `docker-compose.yml` para agregar un tercer servicio (`modelo-podado`) que sirva `modelo_podado.h5` del proyecto 1 (podado pero **sin** cuantizar), y extiende `benchmark/compare.py` para comparar los tres escenarios en una sola tabla. Esto aísla el efecto de cada técnica: ¿cuánto aporta el pruning solo, y cuánto aporta adicionalmente la quantization?
