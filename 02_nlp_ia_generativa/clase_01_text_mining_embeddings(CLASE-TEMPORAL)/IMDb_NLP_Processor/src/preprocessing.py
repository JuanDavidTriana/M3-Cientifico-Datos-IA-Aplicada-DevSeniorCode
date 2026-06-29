"""Pipeline de limpieza de texto para reseñas IMDb."""

from __future__ import annotations

import re
import pandas as pd


_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NUM_RE = re.compile(r"\d+")
_PUNCT_SPECIAL_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Limpia texto en ingles para tareas base de NLP."""
    text = str(text).lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _NUM_RE.sub(" ", text)
    text = _PUNCT_SPECIAL_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame, text_col: str = "review") -> pd.DataFrame:
    """Aplica limpieza sobre un DataFrame y agrega columna review_clean."""
    out = df.copy()
    out["review_clean"] = out[text_col].astype(str).apply(clean_text)
    return out


def show_before_after(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Muestra pares de texto original vs limpio."""
    cols = ["review", "review_clean", "sentiment"]
    return df[cols].head(n)
