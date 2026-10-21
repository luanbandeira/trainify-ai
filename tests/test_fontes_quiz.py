import json

import pytest

from app.graphs.quiz_graph import criar_grafo_quiz, validar_quiz, QuizInvalido
from tests.dubles import LLMFalsa


def resposta(ids):
    return json.dumps({"questions": [{"text": "Quem fornece EPI?", "explanation": "A organização fornece gratuitamente.",
        "source_ids": ids, "sources": [{"excerpt": "Texto inventado"}],
        "options": [{"text": "Organização", "isCorrect": True}, {"text": "Trabalhador", "isCorrect": False}]}]})


@pytest.mark.parametrize("ids", [None, [], ["T999"], [1], "T1", [{"id": "T1"}]])
def test_referencia_ausente_ou_desconhecida_e_recusada(store, ids):
    material = store.curso("curso-a").trechos(lesson_id="nr-06")
    quiz, erros = validar_quiz(resposta(ids), 1, 2, material)
    assert quiz is None and "source_ids" in erros[0]


def test_fontes_vem_do_material_e_nao_do_texto_do_modelo(store):
    material = store.curso("curso-a").trechos(lesson_id="nr-06")
    quiz, erros = validar_quiz(resposta(["T1", "T1"]), 1, 2, material)
    assert not erros
    fontes = quiz["questions"][0]["sources"]
    assert len(fontes) == 1
    assert fontes[0]["excerpt"] == material[0].text
    assert fontes[0]["lesson_id"] == "nr-06"
    assert fontes[0]["source_id"] == material[0].metadata["source_id"]


def test_grafo_corrige_referencia_invalida_em_nova_tentativa(store, embedder):
    llm = LLMFalsa(resposta(["T999"]), resposta(["T1"]))
    estado = criar_grafo_quiz(store, embedder, llm).invoke({
        "course_id": "curso-a", "lesson_id": "nr-06", "num_questions": 1, "num_options": 2,
    })
    assert len(llm.chamadas) == 2
    assert "Fonte T1:" in llm.chamadas[0][1].content
    assert "source_ids" in llm.chamadas[1][-1].content
    assert estado["quiz"]["questions"][0]["sources"][0]["course_id"] == "curso-a"


def test_grafo_nao_publica_quiz_sem_fontes_validas(store, embedder):
    with pytest.raises(QuizInvalido):
        criar_grafo_quiz(store, embedder, LLMFalsa(resposta([]))).invoke({
            "course_id": "curso-a", "lesson_id": "nr-06", "num_questions": 1, "num_options": 2,
        })
