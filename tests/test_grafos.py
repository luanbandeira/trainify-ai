"""Grafos de chat e de quiz, com LLM falsa."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graphs.chat_graph import SEM_CONTEXTO, criar_grafo_chat
from app.graphs.quiz_graph import MaterialInsuficiente, QuizInvalido, criar_grafo_quiz, validar_quiz
from tests.dubles import LLMFalsa


def _chat(store, embedder, llm, **entrada):
    entrada.setdefault("course_ids", ["curso-a"])
    return criar_grafo_chat(store, embedder, llm).invoke(entrada)


def test_chat_monta_prompt_com_trechos_e_historico(store, embedder):
    llm = LLMFalsa("A organização fornece o EPI gratuitamente.")
    historico = [
        {"role": "user", "content": "O que é EPI?"},
        {"role": "assistant", "content": "Equipamento de proteção individual."},
    ]
    saida = _chat(store, embedder, llm, question="Quem deve fornecer o EPI ao trabalhador?", history=historico)

    assert saida["answer"] == "A organização fornece o EPI gratuitamente."
    assert saida["sources"][0]["lesson_id"] == "nr-06"

    mensagens = llm.chamadas[0]
    assert isinstance(mensagens[0], SystemMessage)
    assert "fornecer ao trabalhador, gratuitamente" in mensagens[0].content   # chunk recuperado
    assert [type(m) for m in mensagens[1:]] == [HumanMessage, AIMessage, HumanMessage]  # histórico + pergunta
    assert mensagens[-1].content == "Quem deve fornecer o EPI ao trabalhador?"


def test_pergunta_de_seguimento_usa_a_pergunta_anterior_na_busca(store, embedder):
    llm = LLMFalsa("ok")
    saida = _chat(store, embedder, llm, question="e acima de quantos metros?",
                  history=[{"role": "user", "content": "Quando a norma de trabalho em altura se aplica?"}])
    assert "trabalho em altura" in saida["search_query"]
    assert saida["sources"][0]["lesson_id"] == "nr-35"


def test_sem_contexto_relevante_nao_chama_a_llm(store, embedder):
    llm = LLMFalsa()
    saida = _chat(store, embedder, llm, question="receita de bolo de chocolate")
    assert saida["answer"] == SEM_CONTEXTO
    assert saida["sources"] == []
    assert llm.chamadas == []


def test_aluno_nao_enxerga_curso_fora_da_lista(store, embedder):
    saida = _chat(store, embedder, LLMFalsa(), question="plano estratégico confidencial da empresa")
    assert all(f["course_id"] == "curso-a" for f in saida["sources"])


@pytest.mark.parametrize("papel, trecho", [
    ("STUDENT", "Nunca entregue o gabarito"),
    ("INSTRUCTOR", "sugira perguntas"),
    ("MANAGER", "GESTOR"),
    ("ADMIN", "GESTOR"),
])
def test_prompt_muda_conforme_o_perfil(store, embedder, papel, trecho):
    llm = LLMFalsa()
    _chat(store, embedder, llm, question="Quem fornece o EPI?", role=papel)
    assert trecho in llm.chamadas[0][0].content


def _quiz_json(n=2, opcoes=4, corretas=1):
    return json.dumps({
        "title": "Quiz de EPI",
        "questions": [
            {
                "text": f"Pergunta {i}?",
                "explanation": "Porque sim.",
                "options": [{"text": f"Opção {i}-{j}", "isCorrect": j < corretas} for j in range(opcoes)],
            }
            for i in range(n)
        ],
    })


def test_quiz_valido_no_formato_do_trainify(store, embedder):
    llm = LLMFalsa("```json\n" + _quiz_json() + "\n```")
    estado = criar_grafo_quiz(store, embedder, llm).invoke(
        {"course_id": "curso-a", "lesson_id": "nr-06", "num_questions": 2, "num_options": 4}
    )
    quiz = estado["quiz"]
    assert quiz["title"] == "Quiz de EPI"
    assert quiz["passingScore"] == 70 and quiz["maxAttempts"] == 3
    assert len(quiz["questions"]) == 2
    assert all(sum(o["isCorrect"] for o in q["options"]) == 1 for q in quiz["questions"])
    assert quiz["sources"] == ["nr-06"]
    # o prompt só leva material da aula pedida
    assert "gratuitamente" in llm.chamadas[0][1].content
    assert "Análise de Risco" not in llm.chamadas[0][1].content


def test_quiz_invalido_gera_nova_tentativa_com_os_erros(store, embedder):
    llm = LLMFalsa(_quiz_json(corretas=2), _quiz_json())
    estado = criar_grafo_quiz(store, embedder, llm).invoke(
        {"course_id": "curso-a", "module_id": "m1", "num_questions": 2, "num_options": 4}
    )
    assert len(llm.chamadas) == 2
    assert "exatamente 1 alternativa correta" in llm.chamadas[1][-1].content
    assert len(estado["quiz"]["questions"]) == 2


def test_quiz_desiste_depois_do_limite_de_tentativas(store, embedder):
    with pytest.raises(QuizInvalido):
        criar_grafo_quiz(store, embedder, LLMFalsa("não é JSON")).invoke(
            {"course_id": "curso-a", "lesson_id": "nr-06", "num_questions": 2, "num_options": 4}
        )


def test_quiz_sem_material_indexado(store, embedder):
    with pytest.raises(MaterialInsuficiente):
        criar_grafo_quiz(store, embedder, LLMFalsa()).invoke(
            {"course_id": "curso-a", "lesson_id": "nao-existe", "num_questions": 2, "num_options": 4}
        )


@pytest.mark.parametrize("texto, problema", [
    (_quiz_json(n=1), "foram pedidas 2"),
    (_quiz_json(opcoes=3), "exatamente 4 alternativas"),
    (_quiz_json(corretas=0), "exatamente 1 alternativa correta"),
    ('{"questions": []}', '"questions"'),
])
def test_validacao_do_quiz(texto, problema):
    quiz, erros = validar_quiz(texto, num_questions=2, num_options=4)
    assert quiz is None
    assert any(problema in e for e in erros)


def test_validacao_recusa_todas_as_anteriores():
    dados = json.loads(_quiz_json(n=1))
    dados["questions"][0]["options"][3]["text"] = "Todas as anteriores"
    _, erros = validar_quiz(json.dumps(dados), 1, 4)
    assert erros


def test_resposta_cortada_no_chat_ainda_e_mostrada(store, embedder):
    from app.llm import LLMRespostaCortada

    class LLMCortada(LLMFalsa):
        def gerar(self, mensagens):
            raise LLMRespostaCortada("cortada", "A organização fornece o EPI e")

    saida = _chat(store, embedder, LLMCortada(), question="Quem fornece o EPI ao trabalhador?")
    assert saida["answer"] == "A organização fornece o EPI e (...)"
