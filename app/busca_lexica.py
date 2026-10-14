"""Busca por palavras-chave (BM25), usada junto com a busca vetorial.

Os embeddings entendem o sentido ("cinto para subir no telhado" ≈ trabalho em
altura), mas tratam números e siglas como ruído: "item 35.4.1" e "item 35.5.1"
viram vetores quase iguais. O BM25 faz o contrário: só acha palavras que estão
no texto, mas acha exatamente. Juntar os dois cobre as duas formas de perguntar.
"""

import math
import re
import unicodedata
from collections import Counter

# Itens de norma ("35.4.1") ficam inteiros; o resto vira palavras sem acento
_TOKEN = re.compile(r"\d+(?:\.\d+)+|\w+")

_STOPWORDS = set("""
a ao aos as at com como da das de do dos e ela ele em entre essa esse esta este eu foi ha isso
la mais mas me na nas nao no nos o os ou para pela pelas pelo pelos por qual quais quando que
quem se sem ser seu sua sao so tem um uma umas uns voce
""".split())


def tokenizar(texto: str) -> list[str]:
    sem_acento = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return [t for t in _TOKEN.findall(sem_acento) if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]


class IndiceBM25:
    """BM25 clássico (Okapi) sobre os trechos de um curso, montado em memória."""

    def __init__(self, textos: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(textos)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._tamanhos = []
        for posicao, texto in enumerate(textos):
            contagem = Counter(tokenizar(texto))
            self._tamanhos.append(sum(contagem.values()))
            for termo, frequencia in contagem.items():
                self._postings.setdefault(termo, []).append((posicao, frequencia))
        self._media = (sum(self._tamanhos) / self.n) if self.n else 0.0

    def _idf(self, termo: str) -> float:
        df = len(self._postings.get(termo, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def buscar(self, consulta: str, k: int) -> list[tuple[int, float]]:
        """Devolve (posição do trecho, nota BM25) dos k melhores, só os com nota > 0."""
        notas: dict[int, float] = {}
        for termo in set(tokenizar(consulta)):
            idf = self._idf(termo)
            for posicao, frequencia in self._postings.get(termo, ()):
                normalizacao = 1 - self.b + self.b * self._tamanhos[posicao] / (self._media or 1)
                notas[posicao] = notas.get(posicao, 0.0) + idf * frequencia * (self.k1 + 1) / (
                    frequencia + self.k1 * normalizacao
                )
        return sorted(notas.items(), key=lambda par: par[1], reverse=True)[:k]
