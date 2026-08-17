# Dataset -- dónde debe estar y cómo se usa

## Dataset

**Pediatric Chest X-ray Pneumonia** (Kaggle, usuario `andrewmvd`):
https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray

5,856 radiografías de tórax pediátricas, etiquetadas `NORMAL` o `PNEUMONIA`.

**Cita obligatoria** (incluirla también en el Executive Summary):

> Kermany D, Goldbaum M, Cai W, et al. *Identifying Medical Diagnoses and Treatable
> Diseases by Image-Based Deep Learning*. Cell. 2018;172(5):1122-1131.
> doi:10.1016/j.cell.2018.02.010
>
> Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence
> Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, v2.
> http://dx.doi.org/10.17632/rscbjbr9sj.2

Uso estrictamente académico. **Las imágenes no se suben al repositorio** (`.gitignore`
ya excluye `archive*/` y los `.csv` generados en `data/`); son ~1.2 GB.

## Dónde colocar el dataset

Al día de hoy, el dataset ya está descomprimido un nivel arriba de este proyecto:

```text
clase_04_laboratorio_practico/
├── archive (2)/Pediatric Chest X-ray Pneumonia/{train,test}/{NORMAL,PNEUMONIA}
└── AI-XRay/                    <- este proyecto
```

`src/config.py` ya apunta ahí por defecto (`DEFAULT_DATA_DIR`). No hace falta mover
nada para correr los notebooks tal cual.

Si prefieres tener las imágenes en otro lugar (por ejemplo, si subes este proyecto a tu
propio repositorio y no quieres depender de una ruta relativa "hacia arriba"), colócalas
donde quieras y define la variable de entorno antes de correr los notebooks:

```bash
export AIXRAY_DATA_DIR="/ruta/a/tu/copia/del/dataset"
```

(en Windows/PowerShell: `$env:AIXRAY_DATA_DIR = "C:\ruta\a\tu\copia\del\dataset"`)

Esta carpeta `data/` solo contiene este `README.md`; los archivos `manifest.csv` y
`manifest_split.csv` que ves mencionados abajo se generan automáticamente aquí mismo la
primera vez que corres `01_exploracion.ipynb` -- no hace falta crear nada a mano.

## Composición real (confirmada)

| Carpeta original | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| `train/` | 1,349 | 3,883 | 5,232 |
| `test/` | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

## Por qué este proyecto NO usa el split `train/`/`test/` original tal cual

1. **No trae un set de validación usable.** La versión típica de este dataset separa
   solo 16 imágenes para validación -- insuficiente para monitorear el entrenamiento
   con `EarlyStopping`/`ReduceLROnPlateau`.
2. **Fuga de datos a nivel paciente (confirmada).** Las imágenes de `PNEUMONIA` siguen
   el patrón `personNNNN_{bacteria|virus}_MMMM.jpeg`, donde `NNNN` identifica al
   paciente. Al inspeccionar el dataset se confirmó que el mismo paciente puede tener
   imágenes tanto en `train/` como en `test/` (ejemplo real encontrado: `person100`
   tiene imágenes en ambas carpetas). Entrenar y evaluar con las carpetas originales
   sobreestimaría el desempeño real del modelo.

Por eso `src/dataset.py` junta TODAS las imágenes (train + test originales) y
genera su propio split 70/15/15, agrupado por paciente cuando el paciente es
identificable (ver `split_manifest()` y el chequeo de fuga en
`notebooks/01_exploracion.ipynb`).

**Limitación conocida:** las imágenes `NORMAL` no traen un identificador de paciente
confiable en el nombre de archivo, así que el agrupamiento por paciente solo protege
completamente a la clase `PNEUMONIA`. Esto se documenta también en la sección de
"Limitaciones" del Executive Summary -- es una limitación real del dataset, no un bug
del pipeline.

## Subconjunto usado en la demo de clase

Por rapidez (una sesión de 4-6 horas), la demo/laboratorio de la Clase 4 entrena con un
subconjunto de **2,000 imágenes balanceadas** (1,000 NORMAL + 1,000 PNEUMONIA),
muestreadas con semilla fija (`SEED=42` en `src/config.py`) para que los resultados
sean reproducibles entre estudiantes. El dataset completo (5,856 imágenes, desbalanceado
~2.7:1 a favor de PNEUMONIA) queda disponible como extensión natural del proyecto --
en ese caso, usar `compute_balanced_class_weight()` (`src/train.py`) para
compensar el desbalance en vez de volver a submuestrear.
