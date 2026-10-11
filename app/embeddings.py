"""Etapa 4 do pipeline: embeddings.

Modelo local e multilíngue (intfloat/multilingual-e5-small): a vetorização roda
na própria máquina, sem custo por consulta. A família e5 espera os prefixos
"query: " e "passage: ".
"""

from functools import lru_cache
from typing import Protocol

import numpy as np

from app import config


class Embedder(Protocol):
    dimension: int

    def embed_passages(self, textos: list[str]) -> np.ndarray: ...

    def embed_query(self, texto: str) -> np.ndarray: ...


class E5Embedder:
    def __init__(self, nome_modelo: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self._modelo = SentenceTransformer(nome_modelo, device="cpu")
        self.dimension = self._modelo.get_sentence_embedding_dimension()

    def _encode(self, textos: list[str]) -> np.ndarray:
        vetores = self._modelo.encode(
            textos, batch_size=32, normalize_embeddings=True, show_progress_bar=len(textos) > 200
        )
        return np.asarray(vetores, dtype="float32")

    def embed_passages(self, textos: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in textos])

    def embed_query(self, texto: str) -> np.ndarray:
        return self._encode([f"query: {texto}"])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return E5Embedder()
