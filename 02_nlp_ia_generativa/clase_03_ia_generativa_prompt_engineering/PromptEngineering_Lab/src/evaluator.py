"""Evaluacion simple de respuestas de prompting."""

from __future__ import annotations


def score_response(
    response: str,
    expected_keywords: list[str] | None = None,
    max_length: int = 500,
) -> dict[str, int | float]:
    """Puntua una respuesta con criterios basicos (0 o 1 por criterio)."""
    text = response.strip()
    lower = text.lower()

    scores: dict[str, int | float] = {
        "has_content": 1 if len(text) > 10 else 0,
        "not_too_long": 1 if len(text) < max_length else 0,
        "not_empty": 1 if text else 0,
    }

    if expected_keywords:
        hits = sum(1 for kw in expected_keywords if kw.lower() in lower)
        scores["keyword_hits"] = hits
        scores["keyword_ratio"] = hits / len(expected_keywords) if expected_keywords else 0.0

    scores["total"] = sum(v for k, v in scores.items() if k not in ("keyword_hits", "keyword_ratio"))
    return scores


def compare_strategies(results: list[dict]) -> list[dict]:
    """Agrega scores a una lista de {estrategia, prompt, respuesta}."""
    enriched = []
    for row in results:
        scores = score_response(row.get("respuesta", ""), row.get("keywords"))
        enriched.append({**row, **scores})
    return enriched
