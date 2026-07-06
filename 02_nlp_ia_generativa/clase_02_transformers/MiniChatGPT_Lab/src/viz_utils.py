"""Visualizaciones para el notebook de MiniChatGPT."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def show_temperature_comparison(results: list[dict]) -> None:
    """Tabla clara con respuestas por temperatura (mas legible que barras)."""
    df = pd.DataFrame(
        {
            "Temperatura": [f"T={r['temperature']}" for r in results],
            "Respuesta": [r["reply"] for r in results],
            "Caracteres": [len(r["reply"]) for r in results],
        }
    )
    display(df)


def plot_temperature_comparison(results: list[dict], save_path: str | None = None) -> None:
    """Grafico de longitud + tabla de respuestas."""
    show_temperature_comparison(results)

    fig, ax = plt.subplots(figsize=(8, max(2.5, len(results) * 0.8)))
    temps = [str(r["temperature"]) for r in results]
    lengths = [len(r["reply"]) for r in results]
    colors = ["#3498db", "#2ecc71", "#e74c3c"][: len(results)]

    ax.barh(temps, lengths, color=colors, alpha=0.85)
    ax.set_xlabel("Longitud de la respuesta (caracteres)")
    ax.set_title("Efecto de la temperatura en la respuesta", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_context_usage(tokens_used: int, max_tokens: int, save_path: str | None = None) -> None:
    """Barra de uso de context window."""
    pct = min(tokens_used / max_tokens, 1.0)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh([0], [1.0], color="#ecf0f1", height=0.45, label="Limite total")
    ax.barh([0], [pct], color="#9b59b6", height=0.45, label="Contexto usado")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"Contexto usado: {tokens_used}/{max_tokens} tokens ({pct:.0%})")
    ax.set_title("Context Window del Mini ChatGPT", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
