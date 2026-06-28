"""Funciones de EDA para reseñas de IMDb."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def add_review_length(df: pd.DataFrame, text_col: str = "review") -> pd.DataFrame:
    """Agrega columnas de longitud en caracteres y palabras."""
    out = df.copy()
    out["review_len_chars"] = out[text_col].astype(str).str.len()
    out["review_len_words"] = out[text_col].astype(str).str.split().str.len()
    return out


def basic_report(df: pd.DataFrame) -> dict:
    """Retorna metadatos basicos para inspeccion inicial."""
    return {
        "num_records": int(len(df)),
        "num_columns": int(len(df.columns)),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": df.isna().sum().to_dict(),
    }


def sentiment_distribution(df: pd.DataFrame, label_col: str = "sentiment") -> pd.Series:
    """Distribucion de clases de sentimiento."""
    return df[label_col].value_counts(dropna=False)


def longest_reviews(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Devuelve las reseñas mas largas en palabras."""
    return df.nlargest(top_n, "review_len_words")[["review", "sentiment", "review_len_words"]]


def shortest_reviews(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Devuelve las reseñas mas cortas en palabras."""
    return df.nsmallest(top_n, "review_len_words")[["review", "sentiment", "review_len_words"]]


def plot_sentiment_distribution(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Grafica la distribucion de clases de sentimiento."""
    counts = sentiment_distribution(df)
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
    plt.title("Distribucion de Sentimientos")
    plt.xlabel("Sentimiento")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_length_histogram(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Histograma de longitud de reseñas por numero de palabras."""
    plt.figure(figsize=(7, 4))
    plt.hist(df["review_len_words"], bins=40, color="#2ca02c", alpha=0.8)
    plt.title("Distribucion de Longitud de Resenas (palabras)")
    plt.xlabel("Palabras por reseña")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
