"""Dublês para rodar os testes sem baixar modelo, sem índice real e sem chave de API."""

import hashlib
import re

import numpy as np


class EmbedderFalso:
    """Saco de palavras com hash: textos que dividem palavras ficam próximos."""

    dimension = 256

    def _vetor(self, texto: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype="float32")
        for palavra in re.findall(r"\w{3,}", texto.lower()):
            v[int(hashlib.md5(palavra.encode()).hexdigest(), 16) % self.dimension] += 1
        norma = np.linalg.norm(v)
        return v / norma if norma else v

    def embed_passages(self, textos):
        return np.stack([self._vetor(t) for t in textos]) if textos else np.zeros((0, self.dimension), "float32")

    def embed_query(self, texto):
        return self._vetor(texto)


class LLMFalsa:
    """Devolve as respostas programadas, na ordem, e guarda o que recebeu."""

    def __init__(self, *respostas: str):
        self.respostas = list(respostas) or ["resposta da LLM"]
        self.chamadas = []

    def gerar(self, mensagens):
        self.chamadas.append(mensagens)
        return self.respostas[min(len(self.chamadas), len(self.respostas)) - 1]

    def gerar_stream(self, mensagens):
        """A mesma resposta, entregue palavra por palavra."""
        yield from re.findall(r"\S+\s*", self.gerar(mensagens))
