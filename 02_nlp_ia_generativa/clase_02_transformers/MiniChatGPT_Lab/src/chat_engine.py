"""Motor de chat autoregresivo con GPT-2 (Mini ChatGPT)."""

from __future__ import annotations

from dataclasses import dataclass, field

from prompt_utils import build_prompt, count_tokens, extract_assistant_reply


@dataclass
class ChatConfig:
    """Hiperparametros de generacion visibles en el notebook."""

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
    system_prompt: str = (
        "You are a helpful AI tutor. Answer in 1-2 clear sentences about NLP and Transformers."
    )
    history: list[dict] = field(default_factory=list)
    _model: object = field(default=None, repr=False)
    _tokenizer: object = field(default=None, repr=False)
    _device: str = field(default="cpu", repr=False)

    def load(self) -> None:
        """Carga modelo y tokenizer (sin pipeline, mas estable en notebooks)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self._model.eval()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Llama a chat.load() primero.")
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Llama a chat.load() primero.")
        return self._model

    def reset(self) -> None:
        """Borra el historial de la conversacion."""
        self.history.clear()

    def context_token_count(self, user_message: str = "") -> int:
        """Tokens actuales del contexto (system + historial + mensaje nuevo)."""
        prompt = build_prompt(self.system_prompt, self.history, user_message)
        return count_tokens(prompt, self.tokenizer)

    def _generate(self, prompt: str) -> str:
        """Genera texto token a token con model.generate()."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def chat(self, user_message: str) -> dict:
        """Genera una respuesta y la agrega al historial."""
        prompt = build_prompt(self.system_prompt, self.history, user_message)
        tokens_in = count_tokens(prompt, self.tokenizer)

        if tokens_in > self.config.max_context_tokens:
            return {
                "user": user_message,
                "assistant": (
                    "[Contexto lleno] Reduce el historial con reset() "
                    "o acorta el system prompt."
                ),
                "tokens_in": tokens_in,
                "tokens_out": 0,
                "prompt": prompt,
            }

        full_text = self._generate(prompt)
        reply = extract_assistant_reply(full_text, prompt)

        if not reply:
            reply = "[Sin respuesta util] Prueba reformular la pregunta en ingles."

        tokens_out = count_tokens(reply, self.tokenizer)

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": reply})

        return {
            "user": user_message,
            "assistant": reply,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "prompt": prompt,
        }

    def history_dataframe(self):
        """Historial como DataFrame para visualizar en notebook."""
        import pandas as pd

        rows = []
        for i, turn in enumerate(self.history, start=1):
            rows.append({"turno": i, "rol": turn["role"], "mensaje": turn["content"]})
        return pd.DataFrame(rows)
