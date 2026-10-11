"""Endpoints HTTP, com as dependências pesadas trocadas por dublês."""

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient

from app import config, main
from tests.dubles import LLMFalsa


@pytest.fixture
def llm():
    return LLMFalsa("Resposta com base no material.")


@pytest.fixture
def cliente(store, embedder, llm):
    main.app.dependency_overrides = {
        main.obter_store: lambda: store,
        main.obter_embedder: lambda: embedder,
        main.obter_llm: lambda: llm,
        main.obter_llm_quiz: lambda: llm,
    }
    yield TestClient(main.app)
    main.app.dependency_overrides = {}


def test_health(cliente):
    corpo = cliente.get("/health").json()
    assert corpo["status"] == "ok"
    assert "llm_configured" in corpo


def test_chat(cliente):
    resposta = cliente.post("/api/chat", json={
        "question": "Quem fornece o EPI ao trabalhador?", "role": "STUDENT", "course_ids": ["curso-a"],
        "history": [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "olá"}],
    })
    assert resposta.status_code == 200
    assert resposta.json()["answer"] == "Resposta com base no material."
    assert resposta.json()["sources"][0]["lesson_id"] == "nr-06"


def test_chave_interna_obrigatoria_quando_configurada(cliente, monkeypatch):
    monkeypatch.setattr(config, "INTERNAL_API_KEY", "segredo")
    corpo = {"question": "EPI?", "course_ids": ["curso-a"]}
    assert cliente.post("/api/chat", json=corpo).status_code == 401
    assert cliente.post("/api/chat", json=corpo, headers={"X-Internal-Key": "errada"}).status_code == 401
    assert cliente.post("/api/chat", json=corpo, headers={"X-Internal-Key": "segredo"}).status_code == 200


def test_chat_recusa_course_id_inseguro(cliente):
    resposta = cliente.post("/api/chat", json={"question": "EPI?", "course_ids": ["../outro"]})
    assert resposta.status_code == 422


def test_ingestao_texto_e_listagem(cliente):
    resposta = cliente.post("/api/ingest/lessons/aula-nova", json={
        "course_id": "curso-c", "title": "Aula de extintores", "tenant_id": "t1",
        "content": "<p>O extintor deve ficar em local visível, sinalizado e desobstruído.</p>" * 3,
    })
    assert resposta.status_code == 200
    assert resposta.json()["chunks"] >= 1
    assert "aula-nova" in cliente.get("/api/courses/curso-c/lessons").json()["lessons"]

    assert cliente.delete("/api/ingest/courses/curso-c/lessons/aula-nova").json()["removed_chunks"] >= 1


def _pdf_com_texto(texto: str) -> bytes:
    """PDF mínimo de uma página com texto extraível."""
    conteudo = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET".encode("latin-1")
    objetos = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(conteudo)).encode() + b" >>\nstream\n" + conteudo + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    saida = io.BytesIO(b"%PDF-1.4\n")
    posicoes = []
    for i, obj in enumerate(objetos, start=1):
        posicoes.append(saida.tell())
        saida.write(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    inicio_xref = saida.tell()
    saida.write(f"xref\n0 {len(objetos) + 1}\n0000000000 65535 f \n".encode())
    for p in posicoes:
        saida.write(f"{p:010d} 00000 n \n".encode())
    saida.write(f"trailer\n<< /Size {len(objetos) + 1} /Root 1 0 R >>\nstartxref\n{inicio_xref}\n%%EOF".encode())
    return saida.getvalue()


def test_ingestao_de_pdf_em_base64(cliente):
    pdf = _pdf_com_texto("A sinalizacao de seguranca usa cores para indicar riscos no local de trabalho.")
    resposta = cliente.post("/api/ingest/lessons/aula-pdf", json={
        "course_id": "curso-c", "title": "Sinalização", "pdf_base64": base64.b64encode(pdf).decode(),
    })
    assert resposta.status_code == 200, resposta.text
    assert resposta.json()["chunks"] == 1


def test_ingestao_recusa_url_fora_da_lista(cliente):
    resposta = cliente.post("/api/ingest/lessons/x", json={
        "course_id": "curso-c", "title": "X", "pdf_url": "http://169.254.169.254/latest/meta-data",
    })
    assert resposta.status_code == 422
    assert "não permitido" in resposta.json()["detail"]


def test_ingestao_sem_conteudo(cliente):
    resposta = cliente.post("/api/ingest/lessons/x", json={"course_id": "curso-c", "title": "Vazia"})
    assert resposta.status_code == 422


def test_quiz(cliente, llm):
    llm.respostas = [json.dumps({"title": "Quiz", "questions": [
        {"text": "Quem fornece o EPI?", "explanation": "Item 6.5.1.",
         "options": [{"text": "A organização", "isCorrect": True}, {"text": "O trabalhador", "isCorrect": False}]},
    ]})]
    resposta = cliente.post("/api/quiz", json={
        "course_id": "curso-a", "lesson_id": "nr-06", "num_questions": 1, "num_options": 2,
    })
    assert resposta.status_code == 200, resposta.text
    assert resposta.json()["questions"][0]["text"] == "Quem fornece o EPI?"


def test_quiz_exige_aula_ou_modulo(cliente):
    assert cliente.post("/api/quiz", json={"course_id": "curso-a"}).status_code == 422


def test_quiz_sem_material(cliente):
    resposta = cliente.post("/api/quiz", json={"course_id": "curso-a", "lesson_id": "nada"})
    assert resposta.status_code == 422


def test_llm_fora_do_ar_vira_503(cliente, llm):
    from app.llm import LLMIndisponivel

    def falhar(_):
        raise LLMIndisponivel("sem rede")

    llm.gerar = falhar
    resposta = cliente.post("/api/chat", json={"question": "Quem fornece o EPI?", "course_ids": ["curso-a"]})
    assert resposta.status_code == 503


def test_demo_usa_somente_o_curso_de_demonstracao(cliente, monkeypatch):
    monkeypatch.setattr(config, "DEMO_COURSE_ID", "curso-a")
    resposta = cliente.post("/demo/chat", json={"question": "Plano estratégico confidencial da empresa B"})
    assert resposta.status_code == 200
    assert all(f["course_id"] == "curso-a" for f in resposta.json()["sources"])


def test_demo_desligado(cliente, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", False)
    assert cliente.get("/").status_code == 404
    assert cliente.post("/demo/chat", json={"question": "x"}).status_code == 404


def test_llm_sem_chave_so_falha_quando_e_chamada(monkeypatch):
    from app.llm import LangChainLLM, LLMIndisponivel

    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm = LangChainLLM()  # criar não falha: o caminho sem contexto não precisa da LLM
    with pytest.raises(LLMIndisponivel):
        llm.gerar([])


def test_demo_quiz_sem_escopo_tem_mensagem_legivel(cliente):
    resposta = cliente.post("/demo/quiz", json={})
    assert resposta.status_code == 422
    assert resposta.json()["detail"] == "informe lesson_id ou module_id"


def test_limite_da_llm_vira_429_com_mensagem_clara(cliente, llm):
    from app.llm import LLMLimiteAtingido

    def estourar(_):
        raise LLMLimiteAtingido("429 tokens per minute")

    llm.gerar = estourar
    resposta = cliente.post("/api/chat", json={"question": "Quem fornece o EPI?", "course_ids": ["curso-a"]})
    assert resposta.status_code == 429
    assert "Aguarde" in resposta.json()["detail"]


def test_chat_tira_citacao_do_numero_do_trecho(cliente, llm):
    llm.respostas = ["A organização fornece o EPI (Trecho 2). Veja também [Trechos 1 e 3]."]
    resposta = cliente.post("/api/chat", json={"question": "Quem fornece o EPI?", "course_ids": ["curso-a"]})
    assert resposta.json()["answer"] == "A organização fornece o EPI. Veja também."


def test_remover_curso_inteiro(cliente):
    assert cliente.delete("/api/ingest/courses/curso-a").json()["removed"] is True
    assert cliente.get("/api/courses/curso-a/lessons").json()["lessons"] == {}
    assert cliente.delete("/api/ingest/courses/curso-a").json()["removed"] is False
