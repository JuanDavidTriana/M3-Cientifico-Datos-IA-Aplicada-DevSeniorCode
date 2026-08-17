"""
Construcción del manifiesto del dataset (rutas + etiquetas + metadatos) y splits.

El dataset original ("Pediatric Chest X-ray Pneumonia", Kermany et al. 2018, vía Kaggle)
viene repartido en dos carpetas, `train/` y `test/`, cada una con subcarpetas `NORMAL/` y
`PNEUMONIA/`. Ese split original NO se usa tal cual en este proyecto por dos razones,
explicadas también en el README:

1. No trae un set de validación utilizable (el набор típico de este dataset trae solo
   16 imágenes en `val/`, insuficiente para monitorear el entrenamiento).
2. El nombre de archivo de las imágenes de PNEUMONIA sigue el patrón
   `personNNNN_{bacteria|virus}_MMMM.jpeg`, donde `NNNN` identifica al paciente. Al
   inspeccionar el dataset se confirmó que el MISMO paciente puede aparecer tanto en
   `train/` como en `test/` (ej. `person100` tiene imágenes en ambas carpetas) -- eso es
   fuga de datos (*data leakage*) a nivel paciente si se entrena y evalúa con esas
   carpetas tal cual.

Por eso este módulo junta TODAS las imágenes (train + test originales) en un solo
manifiesto y hace su propio split 70/15/15, agrupando por paciente cuando el paciente es
identificable (clase PNEUMONIA), para minimizar (no eliminar del todo: las imágenes
NORMAL no traen un identificador de paciente confiable) el riesgo de fuga.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLASSES, DATA_DIR, SEED, SPLIT_RATIOS

PATIENT_ID_RE = re.compile(r"^person(\d+)_")


def _extract_patient_id(filename: str, label_name: str) -> str:
    """Extrae un id de paciente del nombre de archivo cuando es posible.

    - PNEUMONIA: `personNNNN_bacteria_MMMM.jpeg` / `personNNNN_virus_MMMM.jpeg` -> "NNNN"
    - NORMAL: el dataset no expone un id de paciente confiable en el nombre
      (`IM-XXXX-0001.jpeg`), así que se usa el propio nombre de archivo como id
      "de un solo uso" -- no agrupa nada, pero mantiene el mismo esquema de columnas.
    """
    if label_name == "PNEUMONIA":
        m = PATIENT_ID_RE.match(filename)
        if m:
            return f"pneumonia_{m.group(1)}"
    return f"normal_{filename}"


def build_manifest(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Recorre `data_dir/{train,test}/{NORMAL,PNEUMONIA}` y arma un DataFrame con
    una fila por imagen: ruta absoluta, etiqueta, id de paciente y carpeta de origen.

    Nota: esta función usa pandas/`pathlib` (recorrido local). La versión pensada para
    "procesamiento masivo" (el paso que conecta esta clase con Databricks) es
    `build_manifest_spark`, más abajo -- aquí, con miles de imágenes, la diferencia de
    rendimiento es mínima, pero en un dataset real de cientos de miles de imágenes
    (como el NIH Chest X-ray14 completo) este recorrido se volvería el cuello de botella.
    """
    rows = []
    for origin in ["train", "test"]:
        for label_idx, label_name in enumerate(CLASSES):
            folder = data_dir / origin / label_name
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.jpeg")):
                rows.append(
                    {
                        "path": str(path),
                        "filename": path.name,
                        "label": label_idx,
                        "label_name": label_name,
                        "patient_id": _extract_patient_id(path.name, label_name),
                        "origin_split": origin,  # split ORIGINAL del dataset, informativo
                    }
                )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise FileNotFoundError(
            f"No se encontraron imágenes en {data_dir}. Revisa AIXRAY_DATA_DIR "
            "o data/README.md para confirmar dónde debe estar el dataset."
        )
    return manifest


def build_manifest_spark(data_dir: Path = DATA_DIR):
    """Equivalente a `build_manifest` usando PySpark, para el bloque de 'procesamiento
    masivo de datos' de la Clase 4. Con 5,856 imágenes esto es simbólico (no necesita un
    cluster), pero deja el código escrito como se escribiría con cientos de miles de
    imágenes: el mismo `spark.read.format("binaryFile")` que se usa aquí para leer y
    listar imágenes es el que se usaría, sin cambios, sobre una carpeta de un Data Lake.

    Devuelve un DataFrame de Spark con el mismo esquema (columnas) que `build_manifest`.
    Requiere PySpark instalado; si no está disponible, usar `build_manifest` (pandas).
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.appName("AIXRay_Manifest").getOrCreate()

    dfs = []
    for origin in ["train", "test"]:
        for label_name in CLASSES:
            folder = str(Path(data_dir) / origin / label_name)
            df = (
                spark.read.format("binaryFile")
                .option("pathGlobFilter", "*.jpeg")
                .load(folder)
                .select(
                    F.col("path"),
                    F.element_at(F.split(F.col("path"), "/"), -1).alias("filename"),
                    F.lit(CLASSES.index(label_name)).alias("label"),
                    F.lit(label_name).alias("label_name"),
                    F.lit(origin).alias("origin_split"),
                    F.col("length").alias("file_size_bytes"),
                )
            )
            dfs.append(df)

    manifest_sdf = dfs[0]
    for df in dfs[1:]:
        manifest_sdf = manifest_sdf.unionByName(df)

    # patient_id se calcula igual que en la versión pandas, con una UDF simple.
    def _pid(filename, label_name):
        return _extract_patient_id(filename, label_name)

    pid_udf = F.udf(_pid)
    manifest_sdf = manifest_sdf.withColumn("patient_id", pid_udf("filename", "label_name"))
    return manifest_sdf


def make_balanced_subset(manifest: pd.DataFrame, per_class: int, seed: int = SEED) -> pd.DataFrame:
    """Toma `per_class` imágenes de cada clase (muestreo aleatorio con semilla fija) para
    el subconjunto de ~2,000 imágenes que se usa en la demo/laboratorio de clase.

    Se muestrea a nivel de PACIENTE cuando es posible (PNEUMONIA), no a nivel de imagen
    suelta, para no romper la agrupación por paciente antes de hacer el split.
    """
    parts = []
    rng = np.random.RandomState(seed)
    for label_idx, label_name in enumerate(CLASSES):
        subset = manifest[manifest["label"] == label_idx]
        patients = subset["patient_id"].unique()
        rng.shuffle(patients)

        chosen_rows = []
        for pid in patients:
            if len(chosen_rows) >= per_class:
                break
            chosen_rows.append(subset[subset["patient_id"] == pid])
        class_subset = pd.concat(chosen_rows).sample(
            n=min(per_class, sum(len(r) for r in chosen_rows)), random_state=seed
        )
        parts.append(class_subset)

    result = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return result


def split_manifest(
    manifest: pd.DataFrame, ratios: dict = SPLIT_RATIOS, seed: int = SEED
) -> pd.DataFrame:
    """Split propio 70/15/15 (train/val/test) agrupado por `patient_id`, estratificado por
    clase. Un mismo paciente nunca queda repartido entre dos splits distintos.

    Devuelve el mismo manifiesto con una columna nueva `split` (train/val/test), en vez
    de tres DataFrames separados, para que sea fácil guardarlo como un solo CSV/tabla.
    """
    rng = np.random.RandomState(seed)
    manifest = manifest.copy()
    manifest["split"] = ""

    for label_idx in manifest["label"].unique():
        class_df = manifest[manifest["label"] == label_idx]
        patients = class_df["patient_id"].unique()
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        train_patients = set(patients[:n_train])
        val_patients = set(patients[n_train : n_train + n_val])
        test_patients = set(patients[n_train + n_val :])

        manifest.loc[
            (manifest["label"] == label_idx) & (manifest["patient_id"].isin(train_patients)),
            "split",
        ] = "train"
        manifest.loc[
            (manifest["label"] == label_idx) & (manifest["patient_id"].isin(val_patients)), "split"
        ] = "val"
        manifest.loc[
            (manifest["label"] == label_idx) & (manifest["patient_id"].isin(test_patients)),
            "split",
        ] = "test"

    return manifest
