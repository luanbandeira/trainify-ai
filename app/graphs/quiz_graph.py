"""Grafo de geração de quiz a partir do material que o instrutor subiu.

    carregar_material → selecionar_trechos → montar_prompt → chamar_llm → validar_quiz
                                                  ↑                            │
                                                  └──── nova tentativa ────────┘ (JSON inválido)

A saída segue o formato do CreateAssessmentRequest do Trainify. Nada é salvo
aqui: o instrutor revisa a sugestão antes de publicar.
"""

import json
import random
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app import config
from app.embeddings import Embedder
from app.llm import LLM
from app.prompts import QUIZ_SISTEMA
from app.schemas import DificuldadeQuiz
from app.vector_store import Resultado, VectorStore

_DIFICULDADES = {
    "BASIC": "Básico: avalie a compreensão de conceitos e informações explícitas do material, com perguntas diretas.",
    "INTERMEDIATE": "Intermediário: avalie a aplicação dos conceitos em situações de trabalho simples, sustentadas pelo material.",
    "ADVANCED": "Avançado: avalie a análise de situações de trabalho e a combinação de informações presentes no material.",
}

_PROIBIDAS = ("todas as anteriores", "nenhuma das anteriores", "todas as alternativas")


class MaterialInsuficiente(ValueError):
    """A aula ou módulo não tem conteúdo indexado para gerar perguntas."""


class QuizInvalido(RuntimeError):
    """A LLM não conseguiu produzir um quiz válido depois de todas as tentativas."""


class QuizState(TypedDict, total=False):
    # entrada
    course_id: str
    lesson_id: str | None
    module_id: str | None
    topic: str | None
    difficulty: DificuldadeQuiz | None
    num_questions: int
    num_options: int
    # intermediários
    material: list[Resultado]
    selected: list[Resultado]
    messages: list[BaseMessage]
    raw_output: str
    errors: list[str]
    attempts: int
    # saída
    quiz: dict


def _criar_carregador(store: VectorStore):
    def carregar_material(state: QuizState) -> QuizState:
        material: list[Resultado] = []
        if store.existe(state["course_id"]):
            material = store.curso(state["course_id"]).trechos(
                lesson_id=state.get("lesson_id"), module_id=state.get("module_id")
            )
        if not material:
            raise MaterialInsuficiente(
                "Não há material indexado para esta aula ou módulo. Envie o conteúdo (texto ou PDF) antes de gerar o quiz."
            )
        return {"material": material, "attempts": 0, "errors": []}

    return carregar_material


def _criar_seletor(embedder: Embedder):
    def selecionar_trechos(state: QuizState) -> QuizState:
        material = state["material"]
        limite = config.QUIZ_MAX_CHUNKS
        if len(material) <= limite:
            return {"selected": material}

        tema = (state.get("topic") or "").strip()
        if tema:
            # Instrutor pediu um foco: pega os trechos mais próximos do tema
            import numpy as np

            vetores = embedder.embed_passages([r.text for r in material])
            notas = vetores @ embedder.embed_query(tema)
            melhores = sorted(np.argsort(-notas)[:limite])
            return {"selected": [material[i] for i in melhores]}

        # Sem foco: trechos espalhados pelo material inteiro, para cobrir todos os assuntos
        passo = len(material) / limite
        return {"selected": [material[int(i * passo)] for i in range(limite)]}

    return selecionar_trechos


def _montar_prompt(state: QuizState) -> QuizState:
    # Sem numerar os trechos, para as explicações citarem a aula e não "trecho 3"
    trechos = "\n\n".join(f"--- {r.text}" for r in state["selected"])
    pedido = f"Crie {state['num_questions']} perguntas"
    if state.get("topic"):
        pedido += f" com foco em: {state['topic']}"
    pedido += f".\n\nTRECHOS DO MATERIAL:\n{trechos}"

    sistema = QUIZ_SISTEMA.format(opcoes=state["num_options"])
    if state.get("difficulty"):
        sistema += "\n\nDIFICULDADE SOLICITADA: " + _DIFICULDADES[state["difficulty"]]
        sistema += ("\nO nível não autoriza acrescentar conhecimentos externos, inventar regras ou criar pegadinhas. "
                    "Se usar uma situação hipotética, deixe isso explícito. A resposta correta e sua explicação "
                    "devem continuar fundamentadas no material; adapte a complexidade ao conteúdo disponível.")
    mensagens: list[BaseMessage] = [
        SystemMessage(content=sistema),
        HumanMessage(content=pedido),
    ]
    if state.get("errors"):
        mensagens.append(HumanMessage(content=(
            "Sua resposta anterior foi recusada pelos seguintes problemas:\n- "
            + "\n- ".join(state["errors"])
            + "\nGere o JSON completo de novo, corrigindo todos eles."
        )))
    return {"messages": mensagens}


def _criar_chamada_llm(llm: LLM):
    def chamar_llm(state: QuizState) -> QuizState:
        return {"raw_output": llm.gerar(state["messages"]), "attempts": state["attempts"] + 1}

    return chamar_llm


def _extrair_json(texto: str) -> dict:
    """Modelos costumam embrulhar o JSON em blocos de código; pegamos o trecho entre chaves."""
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio < 0 or fim <= inicio:
        raise ValueError("a resposta não contém um objeto JSON")
    return json.loads(texto[inicio: fim + 1])


def validar_quiz(texto: str, num_questions: int, num_options: int) -> tuple[dict | None, list[str]]:
    """Confere a estrutura e devolve (quiz normalizado, lista de problemas)."""
    try:
        dados = _extrair_json(texto)
    except (ValueError, json.JSONDecodeError) as erro:
        return None, [f"JSON inválido: {erro}"]

    perguntas = dados.get("questions")
    if not isinstance(perguntas, list) or not perguntas:
        return None, ['o campo "questions" precisa ser uma lista com as perguntas']

    erros: list[str] = []
    normalizadas = []
    for n, p in enumerate(perguntas, start=1):
        texto_pergunta = p.get("text") if isinstance(p, dict) else None
        texto_pergunta = texto_pergunta.strip() if isinstance(texto_pergunta, str) else ""
        opcoes = p.get("options") if isinstance(p, dict) else None
        if not texto_pergunta:
            erros.append(f"pergunta {n} sem enunciado")
            continue
        if not isinstance(opcoes, list) or len(opcoes) != num_options:
            erros.append(f"pergunta {n} precisa de exatamente {num_options} alternativas")
            continue
        textos = [o["text"].strip() if isinstance(o, dict) and isinstance(o.get("text"), str) else "" for o in opcoes]
        corretas = sum(1 for o in opcoes if isinstance(o, dict) and o.get("isCorrect") is True)
        if any(not t for t in textos):
            erros.append(f"pergunta {n} tem alternativa sem texto")
        elif any(type(o.get("isCorrect")) is not bool for o in opcoes):
            erros.append(f"pergunta {n} precisa de isCorrect booleano em todas as alternativas")
        elif any(q["text"].casefold() == texto_pergunta.casefold() for q in normalizadas):
            erros.append(f"pergunta {n} repete um enunciado anterior")
        elif len({t.lower() for t in textos}) != len(textos):
            erros.append(f"pergunta {n} tem alternativas repetidas")
        elif corretas != 1:
            erros.append(f"pergunta {n} precisa de exatamente 1 alternativa correta (tem {corretas})")
        elif any(proibida in t.lower() for t in textos for proibida in _PROIBIDAS):
            erros.append(f'pergunta {n} usa "todas/nenhuma das anteriores"')
        else:
            normalizadas.append({
                "text": texto_pergunta,
                "explanation": str(p.get("explanation", "")).strip(),
                "options": [{"text": t, "isCorrect": o.get("isCorrect") is True} for t, o in zip(textos, opcoes)],
            })

    if len(normalizadas) < num_questions and not erros:
        erros.append(f"foram pedidas {num_questions} perguntas, vieram {len(normalizadas)}")
    if erros:
        return None, erros

    for pergunta in normalizadas[:num_questions]:
        # LLMs tendem a pôr a correta sempre em primeiro; embaralhamos de forma
        # reproduzível (mesma pergunta → mesma ordem).
        random.Random(pergunta["text"]).shuffle(pergunta["options"])

    return {
        "title": str(dados.get("title") or "").strip() or "Avaliação gerada por IA",
        "passingScore": 70,
        "maxAttempts": 3,
        "questions": normalizadas[:num_questions],
    }, []


def _validar(state: QuizState) -> QuizState:
    quiz, erros = validar_quiz(state["raw_output"], state["num_questions"], state["num_options"])
    if quiz:
        quiz["sources"] = sorted({r.metadata.get("lesson_id") for r in state["selected"]} - {None})
        return {"quiz": quiz, "errors": []}
    return {"errors": erros}


def _proximo_passo(state: QuizState) -> str:
    if state.get("quiz"):
        return "fim"
    if state["attempts"] >= config.QUIZ_MAX_ATTEMPTS:
        return "falhou"
    return "tentar_de_novo"


def _falhou(state: QuizState) -> QuizState:
    raise QuizInvalido("Não foi possível gerar um quiz válido agora: " + "; ".join(state["errors"]))


def criar_grafo_quiz(store: VectorStore, embedder: Embedder, llm: LLM):
    grafo = StateGraph(QuizState)
    grafo.add_node("carregar_material", _criar_carregador(store))
    grafo.add_node("selecionar_trechos", _criar_seletor(embedder))
    grafo.add_node("montar_prompt", _montar_prompt)
    grafo.add_node("chamar_llm", _criar_chamada_llm(llm))
    grafo.add_node("validar_quiz", _validar)
    grafo.add_node("falhou", _falhou)

    grafo.add_edge(START, "carregar_material")
    grafo.add_edge("carregar_material", "selecionar_trechos")
    grafo.add_edge("selecionar_trechos", "montar_prompt")
    grafo.add_edge("montar_prompt", "chamar_llm")
    grafo.add_edge("chamar_llm", "validar_quiz")
    grafo.add_conditional_edges(
        "validar_quiz", _proximo_passo, {"fim": END, "tentar_de_novo": "montar_prompt", "falhou": "falhou"}
    )
    grafo.add_edge("falhou", END)
    return grafo.compile()
