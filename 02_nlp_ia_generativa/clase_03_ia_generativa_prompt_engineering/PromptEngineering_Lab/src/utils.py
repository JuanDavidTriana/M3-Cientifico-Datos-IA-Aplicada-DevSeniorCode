"""Utilidades generales del laboratorio."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_lab_root(start: Path | None = None) -> Path:
    start = start or Path.cwd()
    for folder in [start, *start.parents]:
        if (folder / "src").exists() and "PromptEngineering" in folder.name:
            return folder
    raise FileNotFoundError("Abre el notebook desde PromptEngineering_Lab/notebooks/")


def load_dataset(name: str, root: Path | None = None) -> pd.DataFrame:
    root = root or find_lab_root()
    path = root / "datasets" / name
    if not path.exists():
        alt = root.parent / "Clase-03-IA-Generativa-y-Prompt-Engineering" / "datasets" / name
        if alt.exists():
            return pd.read_csv(alt)
        raise FileNotFoundError(f"No se encontro {name} en datasets/")
    return pd.read_csv(path)
