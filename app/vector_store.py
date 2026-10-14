"""Etapa 5 do pipeline: índice vetorial (FAISS).

Cada curso tem o próprio índice em data/indexes/<course_id>/:
  - snapshot.npz: índice FAISS e metadados substituídos juntos, atomicamente

Um índice por curso isola o conteúdo de uma empresa do de outra e deixa simples
reindexar ou apagar uma aula: o curso é pequeno o bastante para reconstruir o
índice inteiro sem custo perceptível.
"""

import json
import os
import copy
import re
import shutil
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
    if not _ID_VALIDO.fullmatch(valor or ""):
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
        snapshot = self.pasta / "snapshot.npz"
        if snapshot.exists():
            with np.load(snapshot, allow_pickle=False) as dados:
                self.index = faiss.deserialize_index(dados["index"])
                self.chunks = json.loads(dados["chunks"].tobytes().decode("utf-8"))
            self._validar()
            return
        arquivo_indice = self.pasta / "index.faiss"
        arquivo_chunks = self.pasta / "chunks.jsonl"
        if not arquivo_indice.exists():
            return
        self.index = faiss.read_index(str(arquivo_indice))
        with arquivo_chunks.open(encoding="utf-8") as f:
            self.chunks = [json.loads(linha) for linha in f if linha.strip()]
        self._validar()

    def _validar(self) -> None:
        if self.index.d != self.dimensao or self.index.ntotal != len(self.chunks):
            raise RuntimeError(f"Índice do curso {self.course_id} está inconsistente; rode a ingestão de novo")

    def salvar(self) -> None:
        self.pasta.mkdir(parents=True, exist_ok=True)
        temporario = self.pasta / "snapshot.npz.tmp"
        try:
            with temporario.open("wb") as arquivo:
                np.savez(
                    arquivo,
                    index=faiss.serialize_index(self.index),
                    chunks=np.frombuffer(json.dumps(self.chunks, ensure_ascii=False).encode("utf-8"), dtype="uint8"),
                )
                arquivo.flush()
                os.fsync(arquivo.fileno())
            temporario.replace(self.pasta / "snapshot.npz")
        finally:
            temporario.unlink(missing_ok=True)

    def copia(self):
        novo = copy.copy(self)
        novo.index = faiss.clone_index(self.index)
        novo.chunks = copy.deepcopy(self.chunks)
        return novo

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
        vetores = np.asarray(vetores, dtype="float32")
        if vetores.shape != (len(chunks), self.dimensao) or not np.isfinite(vetores).all():
            raise ValueError("Os vetores não correspondem aos trechos ou contêm valores inválidos")
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
        pasta = self.pasta_base / validar_id(course_id)
        return self._persistido(pasta) or course_id in self._indices

    @staticmethod
    def _persistido(pasta: Path) -> bool:
        return (pasta / "snapshot.npz").exists() or (pasta / "index.faiss").exists()

    def _salvar(self, course_id: str, indice: CourseIndex) -> None:
        indice.salvar()
        with self._lock_geral:
            self._indices[course_id] = indice

    def indexar_aula(self, course_id: str, lesson_id: str, chunks: list[Chunk], vetores: np.ndarray) -> None:
        with self._lock(course_id):
            indice = self.curso(course_id).copia()
            indice.substituir_aula(lesson_id, chunks, vetores)
            self._salvar(course_id, indice)

    def remover_aula(self, course_id: str, lesson_id: str) -> int:
        if not self.existe(course_id):
            return 0
        with self._lock(course_id):
            indice = self.curso(course_id).copia()
            removidos = indice.remover_aula(lesson_id)
            self._salvar(course_id, indice)
            return removidos

    def cursos_indexados(self) -> list[str]:
        """Cursos com índice em disco. Usado pelo Trainify para perceber que o disco
        foi apagado (ex.: hospedagem gratuita que reinicia) e reindexar."""
        if not self.pasta_base.exists():
            return []
        return sorted(p.name for p in self.pasta_base.iterdir() if self._persistido(p))

    def remover_curso(self, course_id: str) -> bool:
        validar_id(course_id)
        with self._lock(course_id):
            with self._lock_geral:
                self._indices.pop(course_id, None)
            pasta = self.pasta_base / course_id
            if not pasta.exists():
                return False
            shutil.rmtree(pasta)
            return True

    def buscar(self, course_ids: list[str], vetor_consulta: np.ndarray, k: int) -> list[Resultado]:
        """Busca em vários cursos de uma vez (ex.: todos em que o aluno está matriculado)."""
        resultados: list[Resultado] = []
        for course_id in dict.fromkeys(course_ids):
            if self.existe(course_id):
                resultados.extend(self.curso(course_id).buscar(vetor_consulta, k))
        resultados.sort(key=lambda r: r.score, reverse=True)
        return resultados[:k]
