"""Ingestão de uma aula: limpeza → chunking → embeddings → índice do curso.

Chamada pelo Spring sempre que uma aula é criada ou editada, e pelo script que
monta a base de demonstração das NRs.
"""

import hashlib
import json

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from app import config
from app.embeddings import Embedder
from app.text_processing import dividir_em_chunks, extrair_texto_pdf, limpar_paginas, limpar_texto
from app.vector_store import VectorStore, validar_id


class ConteudoVazio(ValueError):
    """A aula não tem texto aproveitável (ex.: PDF escaneado, sem camada de texto)."""


@dataclass
class Aula:
    course_id: str
    lesson_id: str
    title: str
    tenant_id: str | None = None
    module_id: str | None = None
    module_title: str | None = None
    content: str | None = None      # texto escrito pelo instrutor
    transcript: str | None = None   # transcrição do vídeo
    pdf: bytes | None = None        # material em PDF


def baixar_pdf(url: str) -> bytes:
    endereco = urlparse(url)
    host = endereco.hostname or ""
    if endereco.scheme != "https" or not any(host == h or host.endswith("." + h) for h in config.PDF_ALLOWED_HOSTS):
        raise ValueError(f"Endereço de PDF não permitido: {host or url}")

    with httpx.Client(timeout=60, follow_redirects=False) as http, http.stream("GET", url) as resposta:
        resposta.raise_for_status()
        partes, total = [], 0
        for parte in resposta.iter_bytes():
            total += len(parte)
            if total > config.MAX_PDF_BYTES:
                raise ValueError("PDF maior que o limite permitido")
            partes.append(parte)
    conteudo = b"".join(partes)
    if not conteudo.startswith(b"%PDF"):
        raise ValueError("O arquivo baixado não é um PDF")
    return conteudo


def texto_da_aula(aula: Aula) -> str:
    blocos = []
    if aula.content and aula.content.strip():
        blocos.append(limpar_texto(aula.content))
    if aula.pdf:
        blocos.append(limpar_paginas(extrair_texto_pdf(aula.pdf)))
    if aula.transcript and aula.transcript.strip():
        blocos.append("Transcrição do vídeo:\n" + limpar_texto(aula.transcript))
    return "\n\n".join(b for b in blocos if b)


def indexar_aula(aula: Aula, store: VectorStore, embedder: Embedder) -> int:
    """Devolve quantos trechos foram indexados."""
    validar_id(aula.course_id, "course_id")
    validar_id(aula.lesson_id, "lesson_id")

    with store.atualizacao(aula.course_id, aula.lesson_id):
        return _indexar_conteudo(aula, store, embedder)


def _indexar_conteudo(aula: Aula, store: VectorStore, embedder: Embedder) -> int:
    texto = texto_da_aula(aula)
    metadata = {
        "tenant_id": aula.tenant_id,
        "course_id": aula.course_id,
        "module_id": aula.module_id,
        "module_title": aula.module_title,
        "lesson_id": aula.lesson_id,
        "lesson_title": aula.title,
    }
    metadata["version"] = hashlib.sha256(json.dumps({"text": texto, **metadata}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    chunks = dividir_em_chunks(texto, metadata, titulo=aula.title)
    for chunk in chunks:
        chave = f"{aula.course_id}:{aula.lesson_id}:{metadata['version']}:{chunk.metadata['chunk_index']}"
        chunk.metadata["source_id"] = hashlib.sha256(chave.encode()).hexdigest()
    if not chunks:
        raise ConteudoVazio(
            "Não há texto aproveitável nesta aula. Se for um PDF escaneado (imagem), envie uma versão com texto."
        )

    vetores = embedder.embed_passages([c.text for c in chunks])
    store.indexar_aula(aula.course_id, aula.lesson_id, chunks, vetores)
    return len(chunks)
