"""Chat em streaming (Server-Sent Events)."""

import json

import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.graphs.chat_graph import SEM_CONTEXTO, criar_grafo_chat
from app.llm import LLMLimiteAtingido
from tests.dubles import LLMFalsa


@pytest.fixture
def llm():
    return LLMFalsa("A organização fornece o EPI gratuitamente (Trecho 2).")


@pytest.fixture
def cliente(store, embedder, llm):
    main.app.dependency_overrides = {
        main.obter_store: lambda: store,
        main.obter_embedder: lambda: embedder,
        main.obter_llm: lambda: llm,
    }
    yield TestClient(main.app)
    main.app.dependency_overrides = {}


def _eventos(resposta) -> list[tuple[str, object]]:
    eventos = []
    for bloco in resposta.text.strip().split("\n\n"):
        linhas = dict(linha.split(": ", 1) for linha in bloco.splitlines())
        eventos.append((linhas["event"], json.loads(linhas["data"])))
    return eventos


def test_grafo_repassa_os_pedacos_e_devolve_a_resposta_inteira(store, embedder, llm):
    pedacos = []
    saida = criar_grafo_chat(store, embedder, llm, ao_gerar=pedacos.append).invoke(
        {"question": "Quem deve fornecer o EPI ao trabalhador?", "course_ids": ["curso-a"]}
    )
    assert len(pedacos) > 1
    assert "".join(pedacos).strip() == "A organização fornece o EPI gratuitamente (Trecho 2)."
    assert saida["answer"] == "A organização fornece o EPI gratuitamente."  # formatar_resposta continua valendo


def test_api_envia_tokens_e_depois_a_resposta_formatada(cliente):
    resposta = cliente.post("/api/chat/stream", json={"question": "Quem deve fornecer o EPI?", "course_ids": ["curso-a"]})
    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/event-stream")

    eventos = _eventos(resposta)
    tokens = [dados for nome, dados in eventos if nome == "token"]
    nome, final = eventos[-1]
    assert len(tokens) > 1
    assert nome == "done"
    assert final["answer"] == "A organização fornece o EPI gratuitamente."
    assert final["sources"][0]["lesson_id"] == "nr-06"


def test_sem_contexto_nao_chama_a_llm_nem_envia_tokens(cliente, llm):
    resposta = cliente.post("/api/chat/stream", json={"question": "Receita de bolo", "course_ids": ["curso-a"]})
    assert _eventos(resposta) == [("done", {"answer": SEM_CONTEXTO, "sources": []})]
    assert llm.chamadas == []


def test_erro_da_llm_vira_evento_de_erro(cliente, llm):
    def estourar(_):
        raise LLMLimiteAtingido("429")
        yield  # pragma: no cover

    llm.gerar_stream = estourar
    resposta = cliente.post("/api/chat/stream", json={"question": "Quem deve fornecer o EPI?", "course_ids": ["curso-a"]})
    nome, dados = _eventos(resposta)[-1]
    assert nome == "error"
    assert dados["status"] == 429


def test_course_id_inseguro_recusado_antes_do_stream(cliente):
    resposta = cliente.post("/api/chat/stream", json={"question": "EPI?", "course_ids": ["../segredo"]})
    assert resposta.status_code == 422


def test_stream_exige_chave_interna(cliente, monkeypatch):
    monkeypatch.setattr(config, "INTERNAL_API_KEY", "segredo")
    resposta = cliente.post("/api/chat/stream", json={"question": "EPI?", "course_ids": ["curso-a"]})
    assert resposta.status_code == 401


def test_demo_em_stream_so_ve_o_curso_de_demonstracao(cliente, monkeypatch):
    monkeypatch.setattr(config, "DEMO_COURSE_ID", "curso-a")
    resposta = cliente.post("/demo/chat/stream", json={"question": "Plano estratégico confidencial da empresa B"})
    nome, final = _eventos(resposta)[-1]
    assert nome == "done"
    assert all(f["course_id"] == "curso-a" for f in final["sources"])


def test_saturacao_recusa_antes_de_chamar_llm(cliente, llm, monkeypatch):
    import threading

    vagas = threading.BoundedSemaphore(1)
    vagas.acquire()
    monkeypatch.setattr(main, "_vagas_stream", vagas)
    try:
        for rota in ("/api/chat/stream", "/demo/chat/stream"):
            resposta = cliente.post(rota, json={"question": "Quem fornece EPI?", "course_ids": ["curso-a"]})
            assert resposta.status_code == 429
            assert resposta.headers["retry-after"] == "10"
        assert llm.chamadas == []
    finally:
        vagas.release()


def test_erro_libera_vaga_para_proxima_pergunta(cliente, llm, monkeypatch):
    import threading

    monkeypatch.setattr(main, "_vagas_stream", threading.BoundedSemaphore(1))
    def falhar(_):
        raise RuntimeError("falha simulada")
    monkeypatch.setattr(llm, "gerar_stream", falhar)
    pedido = {"question": "Quem deve fornecer o EPI?", "course_ids": ["curso-a"]}
    for _ in range(2):
        resposta = cliente.post("/api/chat/stream", json=pedido)
        assert resposta.status_code == 200
        assert _eventos(resposta)[-1][1]["status"] == 500


def test_desconexao_interrompe_produtor_e_fecha_iterador(store, embedder, llm, monkeypatch):
    import asyncio
    import threading
    from app.schemas import ChatRequest

    fechado = threading.Event()
    produzidos = []
    vagas = threading.BoundedSemaphore(1)
    monkeypatch.setattr(main, "_vagas_stream", vagas)

    def produzir(_):
        try:
            for i in range(10000):
                produzidos.append(i)
                yield "texto "
        finally:
            fechado.set()

    monkeypatch.setattr(llm, "gerar_stream", produzir)
    resposta = main._responder_chat_em_stream(
        ChatRequest(question="Quem fornece EPI?", course_ids=["curso-a"]), store, embedder, llm
    )
    async def consumir():
        await anext(resposta.body_iterator)
        await resposta.body_iterator.aclose()
    asyncio.run(consumir())
    assert fechado.wait(2)
    assert vagas.acquire(timeout=2)
    vagas.release()
    assert len(produzidos) < 10000


def test_cancelamento_fecha_stream_do_provedor():
    from types import SimpleNamespace
    from app.llm import LangChainLLM

    fechados = []
    def produzir(_):
        try:
            yield SimpleNamespace(content="texto", response_metadata={})
            yield SimpleNamespace(content="seguinte", response_metadata={})
        finally:
            fechados.append(True)
    llm = LangChainLLM()
    llm._modelo = SimpleNamespace(stream=produzir)
    fluxo = llm.gerar_stream([])
    assert next(fluxo) == "texto"
    fluxo.close()
    assert fechados == [True]


def test_desconexao_http_mantem_vaga_ate_provedor_retornar(monkeypatch):
    import asyncio
    import threading
    from app.schemas import ChatRequest

    entrou = threading.Event()
    continuar = threading.Event()
    vagas = threading.BoundedSemaphore(1)
    monkeypatch.setattr(main, "_vagas_stream", vagas)

    def responder(*args, ao_gerar):
        entrou.set()
        assert continuar.wait(2)
        ao_gerar("texto")
        raise AssertionError("O processamento deveria ter sido cancelado")

    monkeypatch.setattr(main, "_responder_chat", responder)
    resposta = main._responder_chat_em_stream(
        ChatRequest(question="Pergunta", course_ids=["curso-a"]), None, None, None
    )
    enviados = []
    async def receber():
        return {"type": "http.disconnect"}
    async def enviar(mensagem):
        enviados.append(mensagem)
    async def desconectar():
        await resposta({"type": "http", "asgi": {"spec_version": "2.0"}}, receber, enviar)

    try:
        assert entrou.wait(2)
        asyncio.run(desconectar())
        assert not vagas.acquire(blocking=False)
    finally:
        continuar.set()
    assert vagas.acquire(timeout=2)
    vagas.release()
    assert not any(b"event: done" in m.get("body", b"") for m in enviados)
