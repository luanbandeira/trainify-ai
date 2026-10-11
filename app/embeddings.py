"""Etapa 4 do pipeline: embeddings.

Modelo local e multilíngue (intfloat/multilingual-e5-small): a vetorização roda
na própria máquina, sem custo por consulta. A família e5 espera os prefixos
"query: " e "passage: ".
"""

import threading
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
        # O Trainify manda várias aulas ao mesmo tempo; uma vetorização por vez
        # evita que elas disputem a CPU e deixem todas lentas.
        self._lock = threading.Lock()

    def _encode(self, textos: list[str]) -> np.ndarray:
        with self._lock:
            vetores = self._modelo.encode(
                textos, batch_size=32, normalize_embeddings=True, show_progress_bar=len(textos) > 200
            )
        return np.asarray(vetores, dtype="float32")

    def embed_passages(self, textos: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in textos])

    def embed_query(self, texto: str) -> np.ndarray:
        return self._encode([f"query: {texto}"])[0]


_embedder: Embedder | None = None
_lock_carga = threading.Lock()


def get_embedder() -> Embedder:
    """Carrega o modelo uma única vez. O lock é necessário: requisições
    simultâneas carregando o modelo ao mesmo tempo corrompem os pesos
    ("Cannot copy out of meta tensor")."""
    global _embedder
    if _embedder is None:
        with _lock_carga:
            if _embedder is None:
                _embedder = E5Embedder()
    return _embedder
