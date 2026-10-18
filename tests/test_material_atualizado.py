import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.graphs.chat_graph import SEM_CONTEXTO, _formatar_resposta, criar_grafo_chat
from app.graphs.quiz_graph import MaterialInsuficiente, criar_grafo_quiz
from app.ingestao import Aula, ConteudoVazio, indexar_aula
from app.vector_store import Resultado, VectorStore
from tests.dubles import LLMFalsa
from fastapi.testclient import TestClient
from app import config, main


def test_atualizacao_vazia_persiste_estado_e_nao_consulta_versao_antiga(store, embedder):
    with pytest.raises(ConteudoVazio):
        indexar_aula(Aula(course_id="curso-a", lesson_id="nr-06", title="EPI", content=""), store, embedder)
    reaberto = VectorStore(embedder.dimension, store.pasta_base)
    assert reaberto.situacao("curso-a")["lessons"]["nr-06"]["status"] == "EMPTY"
    assert reaberto.curso("curso-a").trechos(lesson_id="nr-06") == []
    assert all(r.metadata["lesson_id"] != "nr-06" for r in reaberto.buscar(["curso-a"], embedder.embed_query("EPI"), 10))
    with pytest.raises(MaterialInsuficiente):
        criar_grafo_quiz(reaberto, embedder, LLMFalsa()).invoke({"course_id": "curso-a", "lesson_id": "nr-06"})


def test_reenvio_recupera_material_com_nova_versao(store, embedder):
    antiga = store.situacao("curso-a")["lessons"]["nr-06"]["version"]
    aula = Aula(course_id="curso-a", lesson_id="nr-06", title="EPI", content="O kit contém luvas e capacete. " * 10)
    indexar_aula(aula, store, embedder)
    estado = store.situacao("curso-a")["lessons"]["nr-06"]
    assert estado["status"] == "READY" and estado["version"] != antiga
    fonte = store.curso("curso-a").trechos(lesson_id="nr-06")[0].metadata
    indexar_aula(aula, store, embedder)
    assert store.curso("curso-a").trechos(lesson_id="nr-06")[0].metadata["source_id"] == fonte["source_id"]


def test_falha_do_embedder_nao_deixa_material_anterior_disponivel(store, embedder):
    class Indisponivel:
        def embed_passages(self, textos):
            raise RuntimeError("Indisponível")

    with pytest.raises(RuntimeError):
        indexar_aula(Aula(course_id="curso-a", lesson_id="nr-06", title="EPI", content="Material novo sobre acesso corporativo e aprovação do gestor. " * 3), store, Indisponivel())
    assert store.situacao("curso-a")["lessons"]["nr-06"]["status"] == "FAILED"
    assert store.curso("curso-a").trechos(lesson_id="nr-06") == []


def test_processamento_nao_expoe_conteudo_antigo_e_reinicio_e_identificado(store, embedder):
    iniciado, liberar = threading.Event(), threading.Event()

    class Lento:
        def embed_passages(self, textos):
            iniciado.set()
            assert liberar.wait(5)
            return embedder.embed_passages(textos)

    with ThreadPoolExecutor(max_workers=1) as pool:
        futuro = pool.submit(indexar_aula, Aula(course_id="curso-a", lesson_id="nr-06", title="EPI", content="Novo procedimento de fornecimento de equipamentos de proteção. " * 3), store, Lento())
        try:
            assert iniciado.wait(5)
            assert store.situacao("curso-a")["lessons"]["nr-06"]["status"] == "PROCESSING"
            assert store.curso("curso-a").trechos(lesson_id="nr-06") == []
            reaberto = VectorStore(embedder.dimension, store.pasta_base)
            assert reaberto.situacao("curso-a")["lessons"]["nr-06"]["status"] == "INTERRUPTED"
        finally:
            liberar.set()
        futuro.result(timeout=5)
    assert store.situacao("curso-a")["status"] == "READY"


def test_indice_ausente_nao_e_confundido_com_pergunta_fora_do_assunto(store, embedder):
    """Curso sem material nenhum é "não há material" (422 na API), não "pergunta fora
    do assunto": o Trainify mostra a explicação em vez de uma recusa genérica."""
    from app.graphs.quiz_graph import MaterialInsuficiente

    llm = LLMFalsa()
    with pytest.raises(MaterialInsuficiente, match="Ainda não há material"):
        criar_grafo_chat(store, embedder, llm).invoke({"question": "Como funciona?", "course_ids": ["novo"]})
    assert not llm.chamadas


def test_fontes_preservam_trechos_distintos_e_versoes_da_mesma_aula():
    metadata = {"course_id": "curso", "lesson_id": "aula", "version": "v1"}
    contextos = [Resultado("Primeiro", metadata, .9), Resultado("Segundo", metadata, .88)]
    saida = _formatar_resposta({"answer": "Resposta", "contexts": contextos + [contextos[0]]})
    assert [s["excerpt"] for s in saida["sources"]] == ["Primeiro", "Segundo"]
    assert len({s["source_id"] for s in saida["sources"]}) == 2
    assert all(s["kind"] == "retrieved" and s["version"] == "v1" for s in saida["sources"])


def test_recusa_com_material_incompleto_preserva_orientacao():
    saida = _formatar_resposta({
        "answer": "SEM_RESPOSTA_NO_MATERIAL", "contexts": [], "material_warning": "Atualização interrompida.",
    })
    assert saida == {"answer": "Atualização interrompida.", "sources": []}


def test_estado_via_api_exige_chave_e_registra_falha_no_download(store, embedder, monkeypatch):
    import httpx

    monkeypatch.setattr(config, "INTERNAL_API_KEY", "teste")
    main.app.dependency_overrides = {main.obter_store: lambda: store, main.obter_embedder: lambda: embedder}
    try:
        client = TestClient(main.app)
        assert client.get("/api/courses/curso-a/status").status_code == 401
        headers = {"X-Internal-Key": "teste"}
        assert client.get("/api/courses/novo/status", headers=headers).json()["status"] == "MISSING"

        def falhar(url):
            raise httpx.ConnectError("indisponível")

        monkeypatch.setattr(main, "baixar_pdf", falhar)
        resposta = client.post("/api/ingest/lessons/nr-06", headers=headers, json={
            "course_id": "curso-a", "title": "EPI", "pdf_url": "https://docs.google.com/material",
        })
        assert resposta.status_code == 502
        estado = client.get("/api/courses/curso-a/status", headers=headers).json()
        assert estado["lessons"]["nr-06"]["status"] == "FAILED"
        assert store.curso("curso-a").trechos(lesson_id="nr-06") == []
    finally:
        main.app.dependency_overrides = {}


def test_remocao_limpa_estado_e_aviso_parcial_nao_oculta_material_valido(store, embedder):
    with pytest.raises(ConteudoVazio):
        indexar_aula(Aula(course_id="curso-a", lesson_id="nr-35", title="Altura", content=""), store, embedder)
    resposta = criar_grafo_chat(store, embedder, LLMFalsa("A organização fornece EPI.")).invoke({
        "question": "Quem fornece EPI?", "course_ids": ["curso-a"],
    })
    assert "apenas o material disponível" in resposta["answer"]
    assert all(s["lesson_id"] != "nr-35" for s in resposta["sources"])
    store.remover_aula("curso-a", "nr-35")
    assert store.situacao("curso-a")["status"] == "READY"
