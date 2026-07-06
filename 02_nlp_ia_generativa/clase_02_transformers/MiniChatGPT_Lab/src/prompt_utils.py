"""Construccion de prompts estilo ChatGPT para GPT-2."""

from __future__ import annotations

# GPT-2 suele seguir generando turnos falsos; cortamos aqui.
STOP_SEQUENCES = ("\nUser:", "\nSystem:", "\nAssistant:")


def build_prompt(system_prompt: str, history: list[dict], user_message: str) -> str:
    """Arma el texto completo que vera el modelo (system + historial + turno actual)."""
    parts = []
    if system_prompt.strip():
        parts.append(system_prompt.strip())
        parts.append("")

    parts.append("The following is a conversation between a user and an AI assistant about NLP and Transformers.")

    for turn in history:
        role = turn["role"]
        content = turn["content"].strip()
        if role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")

    parts.append(f"User: {user_message.strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def extract_assistant_reply(full_text: str, prompt: str) -> str:
    """Recorta solo la respuesta generada despues del prompt."""
    if full_text.startswith(prompt):
        reply = full_text[len(prompt):]
    elif "Assistant:" in full_text:
        reply = full_text.rsplit("Assistant:", 1)[-1]
    else:
        reply = full_text

    for stop in STOP_SEQUENCES:
        if stop in reply:
            reply = reply.split(stop)[0]

    reply = reply.strip()

    # Quedarse con las primeras oraciones utiles (evita divagar).
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if not lines:
        return ""

    reply = lines[0]
    if len(reply) < 20 and len(lines) > 1:
        reply = f"{lines[0]} {lines[1]}".strip()

    return reply[:400]


def count_tokens(text: str, tokenizer) -> int:
    """Cuenta tokens de un texto."""
    return len(tokenizer.encode(text))
