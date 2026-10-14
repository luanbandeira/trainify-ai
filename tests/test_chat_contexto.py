import pytest

from app import config
from app.graphs.chat_graph import SEM_CONTEXTO, _criar_recuperador, _montar_consulta
from app.vector_store import Resultado
from tests.dubles import LLMFalsa
from tests.test_grafos import _chat


@pytest.mark.parametrize("pergunta", ["Qual é a capital francesa?", "Quem fornece EPI?", "Como solicitar férias?"])
def test_mudanca_de_assunto_nao_herda_consulta_anterior(pergunta):
    resultado = _montar_consulta({"question": pergunta, "history": [{"role": "user", "content": "Trabalho em altura"}]})
    assert resultado["search_query"] == pergunta


def test_preferencia_pela_aula_nao_promove_trecho_irrelevante(embedder, monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.845)
    resultados = [Resultado("irrelevante", {"lesson_id": "atual"}, 0.84), Resultado("relevante", {}, 0.85)]

    class Store:
        def buscar(self, *args, **kwargs):
            return resultados

    estado = _criar_recuperador(Store(), embedder)({"search_query": "EPI", "course_ids": ["a"], "lesson_id": "atual"})
    assert [r.text for r in estado["contexts"]] == ["relevante"]
    assert resultados[0].score == 0.84


def test_recusa_por_falta_de_resposta_nao_apresenta_fontes(store, embedder):
    saida = _chat(store, embedder, LLMFalsa("SEM_RESPOSTA_NO_MATERIAL"), question="Quem fornece o EPI?")
    assert saida["answer"] == SEM_CONTEXTO
    assert saida["sources"] == []
