"""Geração de curso a partir de documentos (LLM falsa, sem rede)."""

import base64
import json

import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.graphs.curso_graph import (
    EstruturaInvalida, MaterialInsuficiente, _paginas, criar_grafo_curso, gerar_com_paciencia, validar_estrutura,
)
from app.llm import LLMLimiteAtingido, LLMRespostaVazia
from app.rascunhos import Rascunhos
from app.vector_store import Resultado
from tests.dubles import LLMFalsa

EPI = ("O empregador deve fornecer o equipamento de proteção individual gratuitamente ao trabalhador. "
       "O trabalhador deve usar o EPI apenas para a finalidade a que se destina e guardar o equipamento. ") * 6
ALTURA = ("Trabalho em altura é toda atividade acima de dois metros do nível inferior com risco de queda. "
          "A análise de risco deve ser feita antes de iniciar o trabalho em altura. ") * 6


def estrutura(docs=("manual-epi.pdf", "altura.pdf"), modulos=2):
    return json.dumps({
        "title": "Segurança no trabalho",
        "description": "Curso sobre EPI e altura.",
        "modules": [
            {"title": f"Módulo {m}", "objective": "Aprender.", "lessons": [
                {"title": "Uso do EPI", "topics": ["fornecimento", "guarda"], "documents": [docs[0]]},
                {"title": "Trabalho em altura", "topics": ["dois metros", "análise de risco"], "documents": [docs[1], "inexistente.pdf"]},
            ]} for m in range(1, modulos + 1)
        ],
    })


@pytest.fixture
def documentos():
    return [{"name": "manual-epi.pdf", "text": EPI}, {"name": "altura.pdf", "text": ALTURA}]


@pytest.fixture(autouse=True)
def _sem_espera(monkeypatch):
    monkeypatch.setattr(config, "CURSO_ESPERA_LIMITE_S", 0)


def _gerar(embedder, documentos, llm_estrutura, llm_aulas, progresso=None, **extra):
    eventos = progresso if progresso is not None else []
    grafo = criar_grafo_curso(embedder, llm_estrutura, llm_aulas, lambda p, e: eventos.append((p, e)))
    return grafo.invoke({"documents": documentos, "min_modules": 1, "max_modules": 4, **extra})


def test_propoe_curso_com_aulas_escritas_a_partir_do_material(embedder, documentos):
    llm_estrutura = LLMFalsa(estrutura())
    llm_aulas = LLMFalsa("## Introdução\nTexto da aula.")
    eventos = []
    proposta = _gerar(embedder, documentos, llm_estrutura, llm_aulas, eventos)["proposal"]

    assert proposta["title"] == "Segurança no trabalho"
    assert len(proposta["modules"]) == 2
    aula_epi, aula_altura = proposta["modules"][0]["lessons"]
    assert aula_epi["content"] == "## Introdução\nTexto da aula."
    # a aula sai dos documentos indicados, e documento inexistente é descartado
    assert [f["document"] for f in aula_epi["sources"]] == ["manual-epi.pdf"]
    assert [f["document"] for f in aula_altura["sources"]] == ["altura.pdf"]
    assert aula_epi["sources"][0]["pages"] == "1"
    # o prompt da aula leva o material certo
    assert "gratuitamente" in llm_aulas.chamadas[0][1].content
    assert "dois metros" in llm_aulas.chamadas[1][1].content
    # o mapa do material (sem LLM) vai para a estrutura, com página
    assert "Documento: manual-epi.pdf" in llm_estrutura.chamadas[0][1].content
    assert "[p. 1]" in llm_estrutura.chamadas[0][1].content
    # progresso só avança
    percentuais = [p for p, _ in eventos if p >= 0]
    assert percentuais == sorted(percentuais)


def test_orientacao_do_instrutor_vai_para_a_estrutura(embedder, documentos):
    llm_estrutura = LLMFalsa(estrutura())
    _gerar(embedder, documentos, llm_estrutura, LLMFalsa("aula"), instructions="Foco em operadores de armazém")
    assert "Foco em operadores de armazém" in llm_estrutura.chamadas[0][1].content


def test_estrutura_invalida_gera_nova_tentativa(embedder, documentos):
    llm_estrutura = LLMFalsa('{"title": "Curso", "modules": []}', estrutura())
    proposta = _gerar(embedder, documentos, llm_estrutura, LLMFalsa("aula"))["proposal"]
    assert len(llm_estrutura.chamadas) == 2
    assert "entre 1 e 4 módulos" in llm_estrutura.chamadas[1][-1].content
    assert proposta["modules"]


def test_desiste_depois_das_tentativas(embedder, documentos):
    with pytest.raises(EstruturaInvalida):
        _gerar(embedder, documentos, LLMFalsa("não é JSON"), LLMFalsa("aula"))


def test_documento_sem_texto_vira_aviso(embedder, documentos):
    documentos.append({"name": "escaneado.pdf", "text": " "})
    proposta = _gerar(embedder, documentos, LLMFalsa(estrutura()), LLMFalsa("aula"))["proposal"]
    assert any("escaneado.pdf" in a for a in proposta["warnings"])


def test_sem_nenhum_texto(embedder):
    with pytest.raises(MaterialInsuficiente):
        _gerar(embedder, [{"name": "vazio.pdf", "text": "curto"}], LLMFalsa(estrutura()), LLMFalsa("aula"))


def test_espera_o_limite_da_ia_em_vez_de_falhar():
    class LLMComLimite(LLMFalsa):
        def gerar(self, mensagens):
            if len(self.chamadas) < 2:
                self.chamadas.append(mensagens)
                raise LLMLimiteAtingido("429")
            return super().gerar(mensagens)

    avisos = []
    llm = LLMComLimite("ok")
    assert gerar_com_paciencia(llm, [], lambda p, e: avisos.append((p, e))) == "ok"
    assert avisos == [(-1, "Aguardando o limite de uso da IA liberar")] * 2


def test_validar_estrutura_limita_modulos():
    _, erros = validar_estrutura(estrutura(modulos=3), ["manual-epi.pdf", "altura.pdf"], 1, 2)
    assert any("entre 1 e 2 módulos" in e for e in erros)


@pytest.mark.parametrize("paginas, esperado", [([1, 2, 3, 7], "1-3, 7"), ([5], "5"), ([2, 4], "2, 4"), ([], "")])
def test_faixas_de_paginas(paginas, esperado):
    assert _paginas([Resultado("x", {"page": p}, 1.0) for p in paginas]) == esperado


# ── gerenciador e API ──

class PoolNaHora:
    """Executa o trabalho na hora, para o teste não depender de threads."""

    def submit(self, funcao, *args):
        funcao(*args)


def test_gerenciador_guarda_resultado_e_erro():
    rascunhos = Rascunhos(lambda pedido, progresso: {"title": pedido["x"]}, pool=PoolNaHora())
    assert rascunhos.criar("r1", {"x": "Curso"})["status"] == "READY"
    assert rascunhos.obter("r1")["result"] == {"title": "Curso"}

    def falhar(pedido, progresso):
        raise MaterialInsuficiente("Nenhum dos documentos tem texto aproveitável.")

    erro = Rascunhos(falhar, pool=PoolNaHora()).criar("r2", {})
    assert erro["status"] == "FAILED"
    assert "texto aproveitável" in erro["error"]


@pytest.fixture
def cliente():
    recebidos = []

    def executar(pedido, progresso):
        recebidos.append(pedido)
        progresso(50, "metade")
        return {"title": "Curso gerado", "modules": []}

    main.app.dependency_overrides = {main.obter_rascunhos: lambda: rascunhos}
    rascunhos = Rascunhos(executar, pool=PoolNaHora())
    cliente = TestClient(main.app)
    cliente.recebidos = recebidos
    yield cliente
    main.app.dependency_overrides = {}


def test_api_cria_e_consulta_rascunho(cliente):
    from tests.test_api import _pdf_com_texto

    pdf = base64.b64encode(_pdf_com_texto("Manual de seguranca da empresa")).decode()
    resposta = cliente.post("/api/course-drafts", json={
        "draft_id": "rascunho-1",
        "documents": [{"name": "manual.pdf", "pdf_base64": pdf}, {"name": "artigo.txt", "text": "Texto do artigo"}],
        "instructions": "Curso curto",
    })
    assert resposta.status_code == 202
    assert cliente.recebidos[0]["documents"][0]["pdf"].startswith(b"%PDF")
    assert cliente.recebidos[0]["instructions"] == "Curso curto"

    estado = cliente.get("/api/course-drafts/rascunho-1").json()
    assert estado["status"] == "READY"
    assert estado["result"]["title"] == "Curso gerado"

    assert cliente.delete("/api/course-drafts/rascunho-1").json()["removed"] is True
    assert cliente.get("/api/course-drafts/rascunho-1").status_code == 404


@pytest.mark.parametrize("documentos, mensagem", [
    ([{"name": "a.pdf", "pdf_base64": "%%%"}], "pdf_base64 inválido"),
    ([{"name": "a.pdf", "pdf_base64": base64.b64encode(b"nao sou pdf").decode()}], "não é um PDF"),
    ([{"name": "a.pdf", "text": "x"}, {"name": "A.PDF", "text": "y"}], "mesmo nome"),
    ([{"name": "a.pdf"}], "não tem conteúdo"),
])
def test_api_recusa_documentos_invalidos(cliente, documentos, mensagem):
    resposta = cliente.post("/api/course-drafts", json={"draft_id": "r", "documents": documentos})
    assert resposta.status_code == 422
    assert mensagem in json.dumps(resposta.json(), ensure_ascii=False)


def test_api_limita_quantidade_de_documentos(cliente, monkeypatch):
    monkeypatch.setattr(config, "CURSO_MAX_DOCUMENTOS", 1)
    resposta = cliente.post("/api/course-drafts", json={
        "draft_id": "r", "documents": [{"name": "a", "text": "x"}, {"name": "b", "text": "y"}],
    })
    assert resposta.status_code == 422


def test_resposta_vazia_e_tentada_de_novo():
    class LLMVaziaUmaVez(LLMFalsa):
        def gerar(self, mensagens):
            if not self.chamadas:
                self.chamadas.append(mensagens)
                raise LLMRespostaVazia("vazia")
            return super().gerar(mensagens)

    assert gerar_com_paciencia(LLMVaziaUmaVez("texto"), []) == "texto"


def test_aula_cortada_e_escrita_de_novo():
    from app.llm import LLMRespostaCortada

    class LLMCortaUmaVez(LLMFalsa):
        def gerar(self, mensagens):
            if not self.chamadas:
                self.chamadas.append(mensagens)
                raise LLMRespostaCortada("cortada", "O programa de treinamento")
            return super().gerar(mensagens)

    assert gerar_com_paciencia(LLMCortaUmaVez("aula completa"), []) == "aula completa"
