"""Utilidades para crear una red minima con Embedding Layer."""

from __future__ import annotations


def build_embedding_model(vocab_size: int, max_len: int, embedding_dim: int = 32):
    """Crea un modelo minimo para visualizar embeddings."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
