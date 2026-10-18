import pytest

from app.graphs.chat_graph import SEM_CONTEXTO, _criar_recuperador, _montar_consulta
from tests.dubles import LLMFalsa
from tests.test_grafos import _chat


@pytest.mark.parametrize("pergunta", ["Qual é a capital francesa?", "Quem fornece EPI?", "Como solicitar férias?"])
def test_mudanca_de_assunto_nao_herda_consulta_anterior(pergunta):
    resultado = _montar_consulta({"question": pergunta, "history": [{"role": "user", "content": "Trabalho em altura"}]})
    assert resultado["search_query"] == pergunta


def test_aula_aberta_filtra_a_busca_em_vez_de_so_priorizar(store, embedder):
    """Com a aula aberta, trechos de outras aulas não concorrem, mesmo que sejam
    mais parecidos com a pergunta (era assim que a NR-1 respondia por dentro da NR-5)."""
    estado = _criar_recuperador(store, embedder)(
        {"search_query": "fornecer EPI gratuitamente ao trabalhador", "course_ids": ["curso-a"], "lesson_id": "nr-35"}
    )
    assert all(r.metadata["lesson_id"] == "nr-35" for r in estado["contexts"])
    # o assunto é de outra aula: ela é apontada, mas o conteúdo dela não entra
    assert estado["other_lessons"] == ["NR-6 EPI"]


def test_recusa_por_falta_de_resposta_nao_apresenta_fontes(store, embedder):
    saida = _chat(store, embedder, LLMFalsa("SEM_RESPOSTA_NO_MATERIAL"), question="Quem fornece o EPI?")
    assert saida["answer"] == SEM_CONTEXTO
    assert saida["sources"] == []
