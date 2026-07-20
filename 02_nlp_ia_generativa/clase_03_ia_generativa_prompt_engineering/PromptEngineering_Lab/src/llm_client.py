"""Cliente LLM en español basado en GPT-2 Spanish."""

from __future__ import annotations

from dataclasses import dataclass, field

MODEL_GENERACION = "datificate/gpt2-small-spanish"
MODEL_SENTIMIENTO = "pysentimiento/robertuito-sentiment-analysis"
MODEL_QA = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"


@dataclass
class LLMConfig:
    model_name: str = MODEL_GENERACION
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.92
    top_k: int = 50
    repetition_penalty: float = 1.1
    seed: int = 42


@dataclass
class LLMClient:
    """Wrapper sobre GPT-2 para experimentos de prompting."""

    config: LLMConfig = field(default_factory=LLMConfig)
    _generator: object = field(default=None, repr=False)

    def load(self) -> None:
        from transformers import pipeline, set_seed

        set_seed(self.config.seed)
        self._generator = pipeline("text-generation", model=self.config.model_name)

    def generate(self, prompt: str, **overrides) -> str:
        if self._generator is None:
            raise RuntimeError("Llama a client.load() primero.")

        max_new = overrides.pop("max_new_tokens", self.config.max_new_tokens)
        temperature = overrides.pop("temperature", self.config.temperature)
        top_p = overrides.pop("top_p", self.config.top_p)
        top_k = overrides.pop("top_k", self.config.top_k)
        rep_penalty = overrides.pop("repetition_penalty", self.config.repetition_penalty)

        out = self._generator(
            prompt,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=rep_penalty,
            pad_token_id=self._generator.tokenizer.eos_token_id,
            **overrides,
        )
        return out[0]["generated_text"]

    def generate_completion(self, prompt: str, **overrides) -> str:
        """Devuelve solo el texto generado despues del prompt."""
        full = self.generate(prompt, **overrides)
        if full.startswith(prompt):
            return full[len(prompt) :].strip()
        return full.strip()
