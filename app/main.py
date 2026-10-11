"""API do trainify-ai.

- /api/*   usada pelo backend Spring do Trainify (exige o cabeçalho X-Internal-Key)
- /demo/*  e /  interface web própria, restrita ao curso de demonstração das NRs
- /health  situação do serviço

Rodar (na raiz do projeto):  uvicorn app.main:app --port 8000
"""

import base64
import binascii
import hmac
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from app import config
from app.embeddings import Embedder, get_embedder
from app.graphs.chat_graph import criar_grafo_chat
from app.graphs.curso_graph import criar_grafo_curso
from app.graphs.quiz_graph import MaterialInsuficiente, QuizInvalido, criar_grafo_quiz
from app.ingestao import Aula, ConteudoVazio, baixar_pdf, indexar_aula
from app.llm import LLM, LLMIndisponivel, LLMLimiteAtingido, get_llm, llm_configurada
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DemoChatRequest,
    DemoQuizRequest,
    IngestRequest,
    IngestResponse,
    QuizRequest,
    QuizResponse,
    RascunhoCursoRequest,
)
from app.rascunhos import Rascunhos
from app.vector_store import VectorStore, validar_id

log = logging.getLogger("trainify-ai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

@asynccontextmanager
async def _ciclo_de_vida(_: FastAPI):
    # Carrega o modelo de embeddings antes de aceitar requisições, para a
    # primeira leva de aulas enviada pelo Trainify não esperar nem disputar a carga.
    if config.PRELOAD_EMBEDDINGS:
        log.info("Carregando o modelo de embeddings %s...", config.EMBEDDING_MODEL)
        obter_store()
        log.info("Modelo carregado.")
    yield


app = FastAPI(
    lifespan=_ciclo_de_vida,
    title="Trainify AI",
    description="Microserviço de RAG do Trainify: chat por perfil e geração de quiz a partir do material dos cursos.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"


# ── Dependências (substituídas por dublês nos testes) ──

def obter_embedder() -> Embedder:
    return get_embedder()


_store: VectorStore | None = None
_lock_store = threading.Lock()


def obter_store() -> VectorStore:
    global _store
    if _store is None:
        with _lock_store:
            if _store is None:
                _store = VectorStore(get_embedder().dimension)
    return _store


def obter_llm() -> LLM:
    return get_llm(config.LLM_MODEL)


def obter_llm_quiz() -> LLM:
    return get_llm(config.QUIZ_LLM_MODEL)


def _gerar_curso(pedido: dict, progresso) -> dict:
    grafo = criar_grafo_curso(
        get_embedder(),
        llm_estrutura=get_llm(config.QUIZ_LLM_MODEL),  # modelo maior: a estrutura pede mais critério
        llm_aulas=get_llm(config.LLM_MODEL),
        progresso=progresso,
    )
    estado = grafo.invoke(pedido)
    return estado["proposal"]


_rascunhos: Rascunhos | None = None


def obter_rascunhos() -> Rascunhos:
    global _rascunhos
    if _rascunhos is None:
        with _lock_store:
            if _rascunhos is None:
                _rascunhos = Rascunhos(_gerar_curso)
    return _rascunhos


def exigir_chave_interna(x_internal_key: str = Header(default="")) -> None:
    if not config.INTERNAL_API_KEY:
        if config.REQUIRE_INTERNAL_KEY:
            raise HTTPException(status_code=503, detail="INTERNAL_API_KEY não configurada neste servidor")
        return
    if not hmac.compare_digest(x_internal_key, config.INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Chave interna inválida")


def exigir_modo_demo() -> None:
    if not config.DEMO_MODE:
        raise HTTPException(status_code=404, detail="Modo de demonstração desligado")


# ── Erros de domínio → respostas HTTP claras ──

@app.exception_handler(LLMLimiteAtingido)
async def _limite_atingido(_: Request, erro: LLMLimiteAtingido):
    log.warning("%s", erro)
    return JSONResponse(
        status_code=429,
        content={"detail": "Muitas perguntas em pouco tempo. Aguarde alguns segundos e tente de novo."},
        headers={"Retry-After": "10"},
    )


@app.exception_handler(LLMIndisponivel)
async def _llm_indisponivel(_: Request, erro: LLMIndisponivel):
    log.error("LLM indisponível: %s", erro)
    return JSONResponse(status_code=503, content={"detail": "O assistente está indisponível no momento. Tente novamente em instantes."})


@app.exception_handler(MaterialInsuficiente)
@app.exception_handler(ConteudoVazio)
async def _sem_material(_: Request, erro: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(erro)})


@app.exception_handler(QuizInvalido)
async def _quiz_invalido(_: Request, erro: QuizInvalido):
    log.warning("%s", erro)
    return JSONResponse(status_code=502, content={"detail": "A IA não conseguiu montar um quiz válido agora. Tente novamente."})


# ── Serviço ──

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": llm_configurada(),
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.LLM_MODEL,
        "quiz_llm_model": config.QUIZ_LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "demo_course_indexed": (config.INDEX_DIR / config.DEMO_COURSE_ID / "index.faiss").exists(),
    }


def _responder_chat(dados: ChatRequest, store: VectorStore, embedder: Embedder, llm: LLM) -> ChatResponse:
    for course_id in dados.course_ids:
        validar_id(course_id)
    grafo = criar_grafo_chat(store, embedder, llm)
    estado = grafo.invoke({
        "question": dados.question,
        "history": [m.model_dump() for m in dados.history],
        "role": dados.role,
        "course_ids": dados.course_ids,
        "lesson_id": dados.lesson_id,
    })
    return ChatResponse(answer=estado["answer"], sources=estado["sources"])


def _gerar_quiz(dados: QuizRequest, store: VectorStore, embedder: Embedder, llm: LLM) -> QuizResponse:
    validar_id(dados.course_id)
    grafo = criar_grafo_quiz(store, embedder, llm)
    estado = grafo.invoke(dados.model_dump())
    return QuizResponse(**estado["quiz"])


def _erro_de_validacao(funcao, *args):
    try:
        return funcao(*args)
    except (MaterialInsuficiente, ConteudoVazio):
        raise
    except ValidationError as erro:
        mensagens = [e["msg"].removeprefix("Value error, ") for e in erro.errors()]
        raise HTTPException(status_code=422, detail="; ".join(mensagens)) from erro
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(exigir_chave_interna)])
def chat(dados: ChatRequest, store=Depends(obter_store), embedder=Depends(obter_embedder), llm=Depends(obter_llm)):
    return _erro_de_validacao(_responder_chat, dados, store, embedder, llm)


@app.post("/api/quiz", response_model=QuizResponse, dependencies=[Depends(exigir_chave_interna)])
def quiz(dados: QuizRequest, store=Depends(obter_store), embedder=Depends(obter_embedder), llm=Depends(obter_llm_quiz)):
    return _erro_de_validacao(_gerar_quiz, dados, store, embedder, llm)


@app.post(
    "/api/ingest/lessons/{lesson_id}", response_model=IngestResponse, dependencies=[Depends(exigir_chave_interna)]
)
def ingerir_aula(lesson_id: str, dados: IngestRequest, store=Depends(obter_store), embedder=Depends(obter_embedder)):
    def executar():
        pdf = None
        if dados.pdf_base64:
            try:
                pdf = base64.b64decode(dados.pdf_base64, validate=True)
            except binascii.Error as erro:
                raise ValueError("pdf_base64 inválido") from erro
            if len(pdf) > config.MAX_PDF_BYTES:
                raise ValueError("PDF maior que o limite permitido")
        elif dados.pdf_url:
            pdf = baixar_pdf(dados.pdf_url)

        aula = Aula(
            course_id=dados.course_id,
            lesson_id=lesson_id,
            title=dados.title,
            tenant_id=dados.tenant_id,
            module_id=dados.module_id,
            module_title=dados.module_title,
            content=dados.content,
            transcript=dados.transcript,
            pdf=pdf,
        )
        chunks = indexar_aula(aula, store, embedder)
        log.info("Aula %s do curso %s indexada (%d chunks)", lesson_id, dados.course_id, chunks)
        return IngestResponse(lesson_id=lesson_id, course_id=dados.course_id, chunks=chunks)

    return _erro_de_validacao(executar)


@app.delete("/api/ingest/courses/{course_id}/lessons/{lesson_id}", dependencies=[Depends(exigir_chave_interna)])
def remover_aula(course_id: str, lesson_id: str, store=Depends(obter_store)):
    removidos = _erro_de_validacao(store.remover_aula, course_id, lesson_id)
    return {"course_id": course_id, "lesson_id": lesson_id, "removed_chunks": removidos}


@app.delete("/api/ingest/courses/{course_id}", dependencies=[Depends(exigir_chave_interna)])
def remover_curso(course_id: str, store=Depends(obter_store)):
    """Curso excluído no Trainify: o índice dele inteiro sai junto."""
    return {"course_id": course_id, "removed": _erro_de_validacao(store.remover_curso, course_id)}


# ── Geração de curso a partir de documentos ──

@app.post("/api/course-drafts", status_code=202, dependencies=[Depends(exigir_chave_interna)])
def criar_rascunho_de_curso(dados: RascunhoCursoRequest, rascunhos=Depends(obter_rascunhos)):
    """Começa a gerar um curso a partir dos documentos. Volta na hora; o andamento
    é consultado em GET /api/course-drafts/{draft_id}."""
    def preparar():
        validar_id(dados.draft_id, "draft_id")
        if len(dados.documents) > config.CURSO_MAX_DOCUMENTOS:
            raise ValueError(f"envie no máximo {config.CURSO_MAX_DOCUMENTOS} documentos")
        documentos = []
        for doc in dados.documents:
            if doc.pdf_base64:
                try:
                    pdf = base64.b64decode(doc.pdf_base64, validate=True)
                except binascii.Error as erro:
                    raise ValueError(f'"{doc.name}": pdf_base64 inválido') from erro
                if len(pdf) > config.MAX_PDF_BYTES:
                    raise ValueError(f'"{doc.name}" é maior que o limite permitido')
                if not pdf.startswith(b"%PDF"):
                    raise ValueError(f'"{doc.name}" não é um PDF')
                documentos.append({"name": doc.name, "pdf": pdf})
            else:
                documentos.append({"name": doc.name, "text": doc.text})
        return {
            "documents": documentos,
            "min_modules": dados.min_modules,
            "max_modules": dados.max_modules,
            "instructions": dados.instructions,
        }

    pedido = _erro_de_validacao(preparar)
    return rascunhos.criar(dados.draft_id, pedido)


@app.get("/api/course-drafts/{draft_id}", dependencies=[Depends(exigir_chave_interna)])
def andamento_do_rascunho(draft_id: str, rascunhos=Depends(obter_rascunhos)):
    estado = rascunhos.obter(draft_id)
    if estado is None:
        raise HTTPException(status_code=404, detail="Rascunho não encontrado (o serviço pode ter reiniciado)")
    return estado


@app.delete("/api/course-drafts/{draft_id}", dependencies=[Depends(exigir_chave_interna)])
def descartar_rascunho(draft_id: str, rascunhos=Depends(obter_rascunhos)):
    return {"draft_id": draft_id, "removed": rascunhos.remover(draft_id)}


@app.get("/api/courses", dependencies=[Depends(exigir_chave_interna)])
def cursos_indexados(store=Depends(obter_store)):
    """Ids dos cursos que têm índice. O Trainify compara com os cursos dele e
    reindexa o que estiver faltando."""
    return {"course_ids": store.cursos_indexados()}


@app.get("/api/courses/{course_id}/lessons", dependencies=[Depends(exigir_chave_interna)])
def aulas_indexadas(course_id: str, store=Depends(obter_store)):
    """Quais aulas do curso já estão indexadas (e com quantos trechos)."""
    if not _erro_de_validacao(store.existe, course_id):
        return {"course_id": course_id, "lessons": {}}
    return {"course_id": course_id, "lessons": store.curso(course_id).aulas()}


# ── Demonstração (disciplina de IA) ──

@app.get("/", include_in_schema=False, dependencies=[Depends(exigir_modo_demo)])
def pagina_inicial():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/demo/course", dependencies=[Depends(exigir_modo_demo)])
def curso_demo():
    return json.loads((config.DATA_DIR / "curso_nrs.json").read_text(encoding="utf-8"))


@app.post("/demo/chat", response_model=ChatResponse, dependencies=[Depends(exigir_modo_demo)])
def chat_demo(dados: DemoChatRequest, store=Depends(obter_store), embedder=Depends(obter_embedder), llm=Depends(obter_llm)):
    pedido = ChatRequest(course_ids=[config.DEMO_COURSE_ID], **dados.model_dump())
    return _erro_de_validacao(_responder_chat, pedido, store, embedder, llm)


@app.post("/demo/quiz", response_model=QuizResponse, dependencies=[Depends(exigir_modo_demo)])
def quiz_demo(dados: DemoQuizRequest, store=Depends(obter_store), embedder=Depends(obter_embedder), llm=Depends(obter_llm_quiz)):
    pedido = _erro_de_validacao(lambda: QuizRequest(course_id=config.DEMO_COURSE_ID, **dados.model_dump()))
    return _erro_de_validacao(_gerar_quiz, pedido, store, embedder, llm)
