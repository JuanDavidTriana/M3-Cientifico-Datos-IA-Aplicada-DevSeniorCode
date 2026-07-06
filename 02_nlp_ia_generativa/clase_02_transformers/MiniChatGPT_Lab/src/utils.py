"""Utilidades generales de MiniChatGPT Lab."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Encuentra la raiz del proyecto buscando carpetas clave."""
    current = start or Path.cwd()
    for folder in [current, *current.parents]:
        if (folder / "src").exists() and (folder / "notebooks").exists():
            name = folder.name.lower()
            if "minichatgpt" in name:
                return folder
    raise FileNotFoundError("No se pudo detectar la raiz de MiniChatGPT_Lab")
