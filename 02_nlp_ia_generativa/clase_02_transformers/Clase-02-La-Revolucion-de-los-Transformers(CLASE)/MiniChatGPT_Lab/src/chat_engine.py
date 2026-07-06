"""Motor de chat autoregresivo con GPT-2 (Mini ChatGPT)."""

from __future__ import annotations

from dataclasses import dataclass, field

from prompt_utils import build_prompt, count_tokens, extract_assistant_reply


@dataclass
class ChatConfig:
    """Hiperparametros de generacion."""
    model_name: str = "gpt2"
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50
    repetition_penalty: float = 1.15
    max_context_tokens: int = 512


@dataclass
class MiniChatGPT:
    """Chatbot minimo basado en decoder-only (GPT-2)."""
    config: ChatConfig = field(default_factory=ChatConfig)
    system_prompt: str = "You are a helpful AI tutor. Answer in 1-2 clear sentences."
    history: list[dict] = field(default_factory=list)
    _model: object = field(default=None, repr=False)
    _tokenizer: object = field(default=None, repr=False)
    _device: str = field(default="cpu", repr=False)

    def load(self) -> None:
        # Escribe tu codigo aqui
        raise NotImplementedError("Completa en clase")

    def reset(self) -> None:
        self.history.clear()

    def context_token_count(self, user_message: str = "") -> int:
        # Escribe tu codigo aqui
        raise NotImplementedError("Completa en clase")

    def chat(self, user_message: str) -> dict:
        # Escribe tu codigo aqui
        raise NotImplementedError("Completa en clase")

    def history_dataframe(self):
        # Escribe tu codigo aqui
        raise NotImplementedError("Completa en clase")
