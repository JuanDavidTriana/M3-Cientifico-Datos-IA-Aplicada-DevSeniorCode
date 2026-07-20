"""Estrategias de prompting en español."""

from __future__ import annotations

import pandas as pd


def zero_shot(tarea: str, entrada: str) -> str:
    return f"{tarea}\nEntrada: {entrada}\nSalida:"


def one_shot(
    tarea: str,
    ejemplo_input: str,
    ejemplo_output: str,
    entrada: str,
) -> str:
    return (
        f"{tarea}\n"
        f"Entrada: {ejemplo_input}\n"
        f"Salida: {ejemplo_output}\n"
        f"Entrada: {entrada}\n"
        f"Salida:"
    )


def few_shot_from_dataframe(
    tarea: str,
    examples_df: pd.DataFrame,
    input_col: str,
    output_col: str,
    entrada: str,
    n_examples: int = 3,
) -> str:
    lineas = [tarea, ""]
    for _, row in examples_df.head(n_examples).iterrows():
        lineas.append(f"Entrada: {row[input_col]}")
        lineas.append(f"Salida: {row[output_col]}")
        lineas.append("")
    lineas.append(f"Entrada: {entrada}")
    lineas.append("Salida:")
    return "\n".join(lineas)


def chain_of_thought(problema: str) -> str:
    return f"{problema}\nPiensa paso a paso. Muestra el razonamiento y luego la respuesta final."


def role_prompt(rol: str, pregunta: str) -> str:
    return f"Eres {rol}.\nPregunta: {pregunta}\nRespuesta:"


def context_prompt(contexto: str, pregunta: str, safe: bool = True) -> str:
    guard = (
        "Usa SOLO el contexto siguiente. Si no encuentras la respuesta, di No disponible.\n"
        if safe
        else ""
    )
    return (
        f"{guard}"
        f"<contexto>\n{contexto}\n</contexto>\n"
        f"Pregunta: {pregunta}\n"
        f"Respuesta:"
    )
