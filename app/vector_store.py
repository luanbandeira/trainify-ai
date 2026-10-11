"""Etapa 5 do pipeline: índice vetorial (FAISS).

Cada curso tem o próprio índice em data/indexes/<course_id>/:
  - index.faiss  vetores normalizados (produto interno = similaridade de cosseno)
  - chunks.jsonl texto e metadados de cada vetor, na mesma ordem do índice

Um índice por curso isola o conteúdo de uma empresa do de outra e deixa simples
reindexar ou apagar uma aula: o curso é pequeno o bastante para reconstruir o
índice inteiro sem custo perceptível.
"""

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app import config
from app.text_processing import Chunk

_ID_VALIDO = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validar_id(valor: str, campo: str = "course_id") -> str:
    """Os ids viram nomes de pasta: só aceitamos caracteres seguros."""
    if not _ID_VALIDO.match(valor or ""):
        raise ValueError(f"{campo} inválido: use letras, números, '-' ou '_'")
    return valor


@dataclass
class Resultado:
    text: str
    metadata: dict
    score: float


class CourseIndex:
    def __init__(self, course_id: str, pasta: Path, dimensao: int):
        self.course_id = course_id
        self.pasta = pasta
        self.dimensao = dimensao
        self.index = faiss.IndexFlatIP(dimensao)
        self.chunks: list[dict] = []
        self._carregar()

    # ── persistência ──

    def _carregar(self) -> None:
        arquivo_indice = self.pasta / "index.faiss"
        arquivo_chunks = self.pasta / "chunks.jsonl"
        if not arquivo_indice.exists():
            return
        self.index = faiss.read_index(str(arquivo_indice))
        with arquivo_chunks.open(encoding="utf-8") as f:
            self.chunks = [json.loads(linha) for linha in f if linha.strip()]
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(f"Índice do curso {self.course_id} está inconsistente; rode a ingestão de novo")

    def salvar(self) -> None:
        self.pasta.mkdir(parents=True, exist_ok=True)
        # Grava em arquivos temporários e troca no fim, para não deixar um índice
        # pela metade se o processo cair no meio da gravação.
        tmp_indice = self.pasta / "index.faiss.tmp"
        tmp_chunks = self.pasta / "chunks.jsonl.tmp"
        faiss.write_index(self.index, str(tmp_indice))
        with tmp_chunks.open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        tmp_indice.replace(self.pasta / "index.faiss")
        tmp_chunks.replace(self.pasta / "chunks.jsonl")

    # ── escrita ──

    def _reconstruir_sem(self, manter: list[bool]) -> None:
        if self.index.ntotal:
            vetores = self.index.reconstruct_n(0, self.index.ntotal)
        else:
            vetores = np.zeros((0, self.dimensao), dtype="float32")
        novo = faiss.IndexFlatIP(self.dimensao)
        selecionados = vetores[np.array(manter, dtype=bool)] if manter else vetores
        if len(selecionados):
            novo.add(selecionados)
        self.index = novo
        self.chunks = [c for c, ok in zip(self.chunks, manter) if ok]

    def remover_aula(self, lesson_id: str) -> int:
        manter = [c["metadata"].get("lesson_id") != lesson_id for c in self.chunks]
        removidos = manter.count(False)
        if removidos:
            self._reconstruir_sem(manter)
        return removidos

    def substituir_aula(self, lesson_id: str, chunks: list[Chunk], vetores: np.ndarray) -> None:
        """Reindexar uma aula editada: sai o conteúdo antigo, entra o novo."""
        self.remover_aula(lesson_id)
        if len(chunks):
            self.index.add(np.asarray(vetores, dtype="float32"))
            self.chunks.extend({"text": c.text, "metadata": c.metadata} for c in chunks)

    # ── leitura ──

    def buscar(self, vetor_consulta: np.ndarray, k: int) -> list[Resultado]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        notas, posicoes = self.index.search(vetor_consulta.reshape(1, -1).astype("float32"), k)
        return [
            Resultado(self.chunks[p]["text"], self.chunks[p]["metadata"], float(n))
            for n, p in zip(notas[0], posicoes[0])
            if p >= 0
        ]

    def trechos(self, lesson_id: str | None = None, module_id: str | None = None) -> list[Resultado]:
        """Todos os trechos de uma aula ou módulo, na ordem do material."""
        selecionados = [
            c for c in self.chunks
            if (lesson_id is None or c["metadata"].get("lesson_id") == lesson_id)
            and (module_id is None or c["metadata"].get("module_id") == module_id)
        ]
        return [Resultado(c["text"], c["metadata"], 1.0) for c in selecionados]

    def aulas(self) -> dict[str, int]:
        contagem: dict[str, int] = {}
        for c in self.chunks:
            lid = c["metadata"].get("lesson_id", "?")
            contagem[lid] = contagem.get(lid, 0) + 1
        return contagem


class VectorStore:
    """Mantém os índices de curso em memória e serializa as escritas por curso."""

    def __init__(self, dimensao: int, pasta_base: Path = config.INDEX_DIR):
        self.dimensao = dimensao
        self.pasta_base = pasta_base
        self._indices: dict[str, CourseIndex] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock_geral = threading.Lock()

    def _lock(self, course_id: str) -> threading.Lock:
        with self._lock_geral:
            return self._locks.setdefault(course_id, threading.Lock())

    def curso(self, course_id: str) -> CourseIndex:
        validar_id(course_id)
        with self._lock_geral:
            if course_id not in self._indices:
                self._indices[course_id] = CourseIndex(course_id, self.pasta_base / course_id, self.dimensao)
            return self._indices[course_id]

    def existe(self, course_id: str) -> bool:
        return (self.pasta_base / validar_id(course_id) / "index.faiss").exists() or course_id in self._indices

    def indexar_aula(self, course_id: str, lesson_id: str, chunks: list[Chunk], vetores: np.ndarray) -> None:
        with self._lock(course_id):
            indice = self.curso(course_id)
            indice.substituir_aula(lesson_id, chunks, vetores)
            indice.salvar()

    def remover_aula(self, course_id: str, lesson_id: str) -> int:
        if not self.existe(course_id):
            return 0
        with self._lock(course_id):
            indice = self.curso(course_id)
            removidos = indice.remover_aula(lesson_id)
            indice.salvar()
            return removidos

    def buscar(self, course_ids: list[str], vetor_consulta: np.ndarray, k: int) -> list[Resultado]:
        """Busca em vários cursos de uma vez (ex.: todos em que o aluno está matriculado)."""
        resultados: list[Resultado] = []
        for course_id in course_ids:
            if self.existe(course_id):
                resultados.extend(self.curso(course_id).buscar(vetor_consulta, k))
        resultados.sort(key=lambda r: r.score, reverse=True)
        return resultados[:k]
