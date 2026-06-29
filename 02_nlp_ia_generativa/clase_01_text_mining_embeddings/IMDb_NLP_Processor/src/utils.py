"""Utilidades generales del proyecto IMDb NLP Processor."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    """Encuentra la raiz del proyecto buscando carpetas clave."""
    current = start or Path.cwd()
    for folder in [current, *current.parents]:
        if (folder / "src").exists() and (folder / "dataset").exists() and (folder / "notebooks").exists():
            return folder
    raise FileNotFoundError("No se pudo detectar la raiz del proyecto IMDb_NLP_Processor")


def load_imdb_dataset(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV de IMDb y valida columnas esperadas."""
    df = pd.read_csv(csv_path)
    expected = {"review", "sentiment"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {sorted(missing)}")
    return df


def sample_dataset(df: pd.DataFrame, n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """Devuelve una muestra estratificada simple para acelerar la practica."""
    if len(df) <= n_samples:
        return df.copy()
    return (
        df.groupby("sentiment", group_keys=False)
        .apply(lambda x: x.sample(n=max(1, int(n_samples / 2)), random_state=random_state))
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def dataset_status(df: pd.DataFrame) -> str:
    """Genera un resumen textual breve del dataset."""
    return (
        f"registros={len(df)}, columnas={len(df.columns)}, "
        f"nulos_total={int(df.isna().sum().sum())}"
    )
