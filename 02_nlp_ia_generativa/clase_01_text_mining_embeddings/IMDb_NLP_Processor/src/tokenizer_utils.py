"""Funciones de tokenizacion y padding con Keras."""

from __future__ import annotations

from collections import Counter


def build_tokenizer(texts, num_words: int = 20000, oov_token: str = "<OOV>"):
    """Construye y ajusta un Tokenizer sobre los textos."""
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_sequences(tokenizer, texts):
    """Convierte textos en secuencias de enteros."""
    return tokenizer.texts_to_sequences(texts)


def top_n_words(tokenizer, n: int = 20):
    """Retorna las palabras mas frecuentes segun word_counts del tokenizer."""
    if not hasattr(tokenizer, "word_counts"):
        return []
    return Counter(tokenizer.word_counts).most_common(n)


def apply_padding(sequences, max_len: int = 200, padding: str = "post", truncating: str = "post"):
    """Aplica pad_sequences a una lista de secuencias."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    return pad_sequences(sequences, maxlen=max_len, padding=padding, truncating=truncating)
