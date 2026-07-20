"""Construccion de prompts estructurados."""

from __future__ import annotations


def build_structured_prompt(
    role: str = "",
    context: str = "",
    instruction: str = "",
    constraints: str = "",
    output_format: str = "",
    examples: str = "",
    user_input: str = "",
) -> str:
    """Arma un prompt con anatomia completa."""
    parts: list[str] = []

    if role.strip():
        parts.append(role.strip())
    if context.strip():
        parts.append(f"<contexto>\n{context.strip()}\n</contexto>")
    if instruction.strip():
        parts.append(f"Tarea: {instruction.strip()}")
    if constraints.strip():
        parts.append(f"Reglas: {constraints.strip()}")
    if output_format.strip():
        parts.append(f"Formato de salida: {output_format.strip()}")
    if examples.strip():
        parts.append(f"Ejemplos:\n{examples.strip()}")
    if user_input.strip():
        parts.append(f"Entrada: {user_input.strip()}")

    parts.append("Respuesta:")
    return "\n\n".join(parts)


STANDARD_CONSTRAINTS = """Reglas:
- Responde SOLO con el contexto proporcionado cuando hay contexto
- Si falta informacion, di 'No disponible'
- No inventes cifras, fechas ni nombres
- Maximo 100 palabras
- Tono profesional"""
