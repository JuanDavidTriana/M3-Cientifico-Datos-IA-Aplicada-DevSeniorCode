"""Construccion de prompts estilo ChatGPT para GPT-2."""

from __future__ import annotations

STOP_SEQUENCES = ("\nUser:", "\nSystem:", "\nAssistant:")


def build_prompt(system_prompt: str, history: list[dict], user_message: str) -> str:
    """Arma el prompt completo."""
    # Escribe tu codigo aqui
    raise NotImplementedError("Completa en clase")


def extract_assistant_reply(full_text: str, prompt: str) -> str:
    """Recorta la respuesta del asistente."""
    # Escribe tu codigo aqui
    raise NotImplementedError("Completa en clase")


def count_tokens(text: str, tokenizer) -> int:
    """Cuenta tokens."""
    # Escribe tu codigo aqui
    raise NotImplementedError("Completa en clase")
