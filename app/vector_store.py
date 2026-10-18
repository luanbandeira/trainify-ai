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
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from app import config
from app.busca_lexica import IndiceBM25
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
    score: float  # similaridade de cosseno: é ela que passa pelo corte do MIN_SCORE
    # Ordem da busca híbrida (cosseno + bônus das palavras-chave); sem BM25, igual ao score
    relevancia: float = field(default=None)

    def __post_init__(self):
        if self.relevancia is None:
            self.relevancia = self.score


class CourseIndex:
    def __init__(self, course_id: str, pasta: Path, dimensao: int):
        self.course_id = course_id
        self.pasta = pasta
        self.dimensao = dimensao
        self.index = faiss.IndexFlatIP(dimensao)
        self.chunks: list[dict] = []
        self.estados: dict[str, dict] = {}
        self._bm25: IndiceBM25 | None = None  # montado na primeira busca, refeito quando os trechos mudam
        self._carregar()

    # ── persistência ──

    def _carregar(self) -> None:
        snapshot = self.pasta / "snapshot.npz"
        if snapshot.exists():
            with np.load(snapshot, allow_pickle=False) as dados:
                self.index = faiss.deserialize_index(dados["index"])
                self.chunks = json.loads(dados["chunks"].tobytes().decode("utf-8"))
                if "lessons" in dados:
                    self.estados = json.loads(dados["lessons"].tobytes().decode("utf-8"))
                for estado in self.estados.values():
                    if estado["status"] == "PROCESSING":
                        estado["status"] = "INTERRUPTED"
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
                    lessons=np.frombuffer(json.dumps(self.estados).encode("utf-8"), dtype="uint8"),
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
        novo.estados = copy.deepcopy(self.estados)
        novo._bm25 = None
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
        self._bm25 = None

    def remover_aula(self, lesson_id: str) -> int:
        self.estados.pop(lesson_id, None)
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
            self._bm25 = None

    # ── leitura ──

    def buscar(self, vetor_consulta: np.ndarray, k: int, texto_consulta: str | None = None) -> list[Resultado]:
        """Busca vetorial. Com `texto_consulta` (e HYBRID_SEARCH ligado), vira busca
        híbrida: os k melhores do BM25 entram como candidatos junto com os k melhores
        por vetor, e a nota BM25 (normalizada de 0 a 1) soma um bônus de até
        HYBRID_WEIGHT na ordem. O corte por MIN_SCORE continua sendo feito sobre o
        cosseno, para a recusa de perguntas fora do assunto não mudar."""
        if self.index.ntotal == 0:
            return []
        if any(e["status"] != "READY" for e in self.estados.values()):
            disponivel = self.copia()
            disponivel._reconstruir_sem([
                self.estados.get(c["metadata"].get("lesson_id"), {}).get("status", "READY") == "READY"
                for c in self.chunks
            ])
            disponivel.estados = {}
            return disponivel.buscar(vetor_consulta, k, texto_consulta)
        k = min(k, self.index.ntotal)
        consulta = vetor_consulta.reshape(1, -1).astype("float32")
        notas, posicoes = self.index.search(consulta, k)
        cossenos = {int(p): float(n) for n, p in zip(notas[0], posicoes[0]) if p >= 0}

        bonus: dict[int, float] = {}
        if texto_consulta and config.HYBRID_SEARCH and config.HYBRID_WEIGHT > 0:
            if self._bm25 is None:
                self._bm25 = IndiceBM25([c["text"] for c in self.chunks])
            lexicos = self._bm25.buscar(texto_consulta, k)
            if lexicos:
                maior = lexicos[0][1]
                bonus = {p: config.HYBRID_WEIGHT * nota / maior for p, nota in lexicos}
                for p in bonus.keys() - cossenos.keys():
                    cossenos[p] = float(self.index.reconstruct(p) @ consulta[0])

        resultados = [
            Resultado(self.chunks[p]["text"], self.chunks[p]["metadata"], nota, nota + bonus.get(p, 0.0))
            for p, nota in cossenos.items()
        ]
        resultados.sort(key=lambda r: r.relevancia, reverse=True)
        return resultados[:k]

    def buscar_na_aula(self, lesson_id: str, vetor_consulta: np.ndarray, k: int,
                       texto_consulta: str | None = None) -> list[Resultado]:
        """Busca só entre os trechos de uma aula. As outras aulas não concorrem:
        com a aula aberta, um trecho de outra aula nunca pode vencer a disputa."""
        if self.estados.get(lesson_id, {}).get("status", "READY") != "READY":
            return []
        posicoes = [i for i, c in enumerate(self.chunks) if c["metadata"].get("lesson_id") == lesson_id]
        if not posicoes:
            return []
        consulta = vetor_consulta.reshape(-1).astype("float32")
        vetores = np.stack([self.index.reconstruct(p) for p in posicoes])
        cossenos = vetores @ consulta

        bonus = np.zeros(len(posicoes), dtype="float32")
        if texto_consulta and config.HYBRID_SEARCH and config.HYBRID_WEIGHT > 0:
            lexicos = IndiceBM25([self.chunks[p]["text"] for p in posicoes]).buscar(texto_consulta, len(posicoes))
            if lexicos:
                maior = lexicos[0][1]
                for local, nota in lexicos:
                    bonus[local] = config.HYBRID_WEIGHT * nota / maior

        resultados = [
            Resultado(self.chunks[p]["text"], self.chunks[p]["metadata"], float(cossenos[i]), float(cossenos[i] + bonus[i]))
            for i, p in enumerate(posicoes)
        ]
        resultados.sort(key=lambda r: r.relevancia, reverse=True)
        return resultados[:k]

    def tem_material(self, lesson_id: str | None = None) -> bool:
        """Há ao menos um trecho pronto para consulta (no curso ou na aula)."""
        return any(
            self.estados.get(c["metadata"].get("lesson_id"), {}).get("status", "READY") == "READY"
            and (lesson_id is None or c["metadata"].get("lesson_id") == lesson_id)
            for c in self.chunks
        )

    def trechos(self, lesson_id: str | None = None, module_id: str | None = None) -> list[Resultado]:
        """Todos os trechos de uma aula ou módulo, na ordem do material."""
        selecionados = [
            c for c in self.chunks
            if (lesson_id is None or c["metadata"].get("lesson_id") == lesson_id)
            and (module_id is None or c["metadata"].get("module_id") == module_id)
            and self.estados.get(c["metadata"].get("lesson_id"), {}).get("status", "READY") == "READY"
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
            return self._locks.setdefault(course_id, threading.RLock())

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
            indice.estados[lesson_id] = {
                "status": "READY", "updated_at": datetime.now(timezone.utc).isoformat(),
                "version": chunks[0].metadata.get("version") if chunks else None,
            }
            self._salvar(course_id, indice)

    @contextmanager
    def atualizacao(self, course_id: str, lesson_id: str):
        validar_id(course_id)
        validar_id(lesson_id, "lesson_id")
        with self._lock(course_id):
            self._marcar(course_id, lesson_id, "PROCESSING")
            try:
                yield
            except Exception as erro:
                from app.ingestao import ConteudoVazio
                self._marcar(course_id, lesson_id, "EMPTY" if isinstance(erro, ConteudoVazio) else "FAILED")
                raise

    def _marcar(self, course_id: str, lesson_id: str, status: str):
        indice = self.curso(course_id).copia()
        anterior = indice.estados.get(lesson_id, {})
        indice.estados[lesson_id] = {
            **anterior, "status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._salvar(course_id, indice)

    def situacao(self, course_id: str) -> dict:
        if not self.existe(course_id):
            return {"course_id": course_id, "status": "MISSING", "lessons": {}}
        indice = self.curso(course_id)
        estados = {lid: {"status": "READY", "version": None} for lid in indice.aulas()}
        estados.update(copy.deepcopy(indice.estados))
        status = "EMPTY" if not estados else "READY" if all(e["status"] == "READY" for e in estados.values()) else "UNAVAILABLE"
        return {"course_id": course_id, "status": status, "lessons": estados}

    def remover_aula(self, course_id: str, lesson_id: str) -> int:
        if not self.existe(course_id):
            return 0
        with self._lock(course_id):
            indice = self.curso(course_id).copia()
            removidos = indice.remover_aula(lesson_id)
            self._salvar(course_id, indice)
            return removidos

    def cursos_indexados(self) -> list[str]:
        """Cursos com ao menos um trecho pronto para consulta. O Trainify compara esta
        lista com os cursos dele e reindexa o que faltar: por isso um curso registrado
        mas vazio (ingestão que falhou, aulas sem texto, índice recriado) fica de fora,
        para ser reindexado em vez de ficar mudo sem ninguém perceber."""
        if not self.pasta_base.exists():
            return []
        cursos = []
        for pasta in sorted(self.pasta_base.iterdir()):
            if not self._persistido(pasta):
                continue
            try:
                if self.curso(pasta.name).tem_material():
                    cursos.append(pasta.name)
            except (ValueError, RuntimeError, OSError):
                continue  # pasta com nome inválido ou índice ilegível: tratar como ausente
        return cursos

    def tem_material(self, course_ids: list[str], lesson_id: str | None = None) -> bool:
        return any(self.existe(c) and self.curso(c).tem_material(lesson_id) for c in dict.fromkeys(course_ids))

    def curso_da_aula(self, course_ids: list[str], lesson_id: str) -> str | None:
        """Em qual dos cursos permitidos a aula está indexada (com ou sem material pronto)."""
        for course_id in dict.fromkeys(course_ids):
            if self.existe(course_id):
                indice = self.curso(course_id)
                if lesson_id in indice.estados or any(c["metadata"].get("lesson_id") == lesson_id for c in indice.chunks):
                    return course_id
        return None

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

    def buscar(
        self, course_ids: list[str], vetor_consulta: np.ndarray, k: int, texto_consulta: str | None = None,
        lesson_id: str | None = None,
    ) -> list[Resultado]:
        """Busca em vários cursos de uma vez (ex.: todos em que o aluno está matriculado).
        Com `lesson_id`, só os trechos daquela aula entram na disputa."""
        resultados: list[Resultado] = []
        for course_id in dict.fromkeys(course_ids):
            if self.existe(course_id):
                indice = self.curso(course_id)
                if lesson_id:
                    resultados.extend(indice.buscar_na_aula(lesson_id, vetor_consulta, k, texto_consulta))
                else:
                    resultados.extend(indice.buscar(vetor_consulta, k, texto_consulta))
        resultados.sort(key=lambda r: r.relevancia, reverse=True)
        return resultados[:k]
