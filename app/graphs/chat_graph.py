"""Grafo de RAG do chat, orquestrado com LangGraph.

    receber_pergunta → montar_consulta → recuperar_contexto → montar_prompt → chamar_llm → formatar_resposta

Só o nó chamar_llm usa a LLM externa. Perguntas de seguimento ("e para altura
acima de 2 m?") são resolvidas sem LLM: a consulta de busca junta a pergunta
atual com a última pergunta do usuário quando há indicação de continuidade.
"""

import re
import hashlib
from collections.abc import Callable
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app import config
from app.embeddings import Embedder
from app.llm import LLM, LLMRespostaCortada
from app.prompts import sistema_para
from app.vector_store import Resultado, VectorStore


class Mensagem(TypedDict):
    role: str      # "user" ou "assistant"
    content: str


class ChatState(TypedDict, total=False):
    # entrada
    question: str
    history: list[Mensagem]
    role: str
    course_ids: list[str]
    lesson_id: str | None
    # intermediários
    search_query: str
    contexts: list[Resultado]
    material_warning: str
    messages: list[BaseMessage]
    # saída
    answer: str
    sources: list[dict]


_SEGUIMENTO = re.compile(
    r"^(e|mas|então|entao|tamb[eé]m|isso|essa|esse|ela|ele|como assim|explique melhor)\b",
    re.IGNORECASE,
)

_CITACAO_DE_TRECHO = re.compile(r"\s*[\(\[]\s*Trechos?\s+\d+(\s*(,|e)\s*\d+)*\s*[\)\]]", re.IGNORECASE)

SEM_CONTEXTO = (
    "Não encontrei esse assunto no material dos seus cursos. Tente perguntar de outro jeito "
    "ou com termos que aparecem nas aulas. Se a dúvida continuar, fale com o instrutor."
)


def _receber_pergunta(state: ChatState) -> ChatState:
    pergunta = re.sub(r"\s+", " ", state.get("question", "")).strip()
    if not pergunta:
        raise ValueError("A pergunta está vazia.")
    historico = [m for m in state.get("history") or [] if m.get("content", "").strip()]
    return {
        "question": pergunta[:2000],
        "history": historico[-config.HISTORY_MESSAGES:],
        "role": (state.get("role") or "STUDENT").upper(),
    }


def _montar_consulta(state: ChatState) -> ChatState:
    pergunta = state["question"]
    anteriores = [m["content"] for m in state["history"] if m["role"] == "user"]
    depende_do_historico = _SEGUIMENTO.match(pergunta)
    if anteriores and depende_do_historico:
        return {"search_query": f"{anteriores[-1]} {pergunta}"}
    return {"search_query": pergunta}


def _criar_recuperador(store: VectorStore, embedder: Embedder):
    def recuperar_contexto(state: ChatState) -> ChatState:
        consulta = state["search_query"]
        vetor = embedder.embed_query(consulta)
        # Busca híbrida: vetor (sentido) + BM25 (palavras exatas, como "item 35.4.1")
        candidatos = store.buscar(state.get("course_ids") or [], vetor, k=config.TOP_K * 2, texto_consulta=consulta)
        relevantes = [r for r in candidatos if r.score >= config.MIN_SCORE]

        # Se o aluno está numa aula, os trechos dela ganham prioridade no desempate
        aula = state.get("lesson_id")
        if aula:
            relevantes.sort(
                key=lambda r: r.relevancia + (0.02 if r.metadata.get("lesson_id") == aula else 0),
                reverse=True,
            )

        situacoes = [store.situacao(c) for c in state.get("course_ids") or []]
        incompletos = [s for s in situacoes if s["status"] != "READY"]
        aviso = ""
        if incompletos:
            status = {e["status"] for s in incompletos for e in s["lessons"].values()}
            if "PROCESSING" in status:
                aviso = "Há material sendo atualizado. Tente novamente após o processamento."
            elif status & {"FAILED", "INTERRUPTED"}:
                aviso = "Há material com atualização interrompida ou que falhou. Peça ao instrutor para reenviá-lo."
            elif "EMPTY" in status or any(s["status"] == "EMPTY" for s in incompletos):
                aviso = "Há material sem texto disponível para consulta. Peça ao instrutor uma versão com texto."
            else:
                aviso = "Há cursos cujo material ainda não está disponível para consulta. Peça ao instrutor para verificar o envio."
        return {"contexts": relevantes[:config.TOP_K], "material_warning": aviso}

    return recuperar_contexto


def _montar_prompt(state: ChatState) -> ChatState:
    trechos = "\n\n".join(
        # Sem numeração de propósito: com "Trecho 3" no prompt, o modelo citava o
        # número (que não significa nada para quem lê) em vez do nome da aula.
        f'--- Da aula "{r.metadata.get("lesson_title") or r.metadata.get("lesson_id")}":\n{r.text}'
        for r in state["contexts"]
    )
    sistema = sistema_para(state["role"]) + f"\n\nTRECHOS DO MATERIAL:\n{trechos}"

    mensagens: list[BaseMessage] = [SystemMessage(content=sistema)]
    for m in state["history"]:
        classe = HumanMessage if m["role"] == "user" else AIMessage
        mensagens.append(classe(content=m["content"]))
    mensagens.append(HumanMessage(content=state["question"]))
    return {"messages": mensagens}


def _criar_chamada_llm(llm: LLM, ao_gerar: Callable[[str], None] | None = None):
    """Com `ao_gerar`, a resposta é pedida em streaming e cada pedaço é repassado
    assim que chega (a interface mostra o texto sendo escrito). O nó devolve a
    resposta inteira do mesmo jeito, para formatar_resposta tratar como sempre."""
    def chamar_llm(state: ChatState) -> ChatState:
        try:
            if ao_gerar and hasattr(llm, "gerar_stream"):
                partes = []
                fluxo = llm.gerar_stream(state["messages"])
                try:
                    for parte in fluxo:
                        partes.append(parte)
                        ao_gerar(parte)
                finally:
                    fechar = getattr(fluxo, "close", None)
                    if fechar:
                        fechar()
                return {"answer": "".join(partes).strip()}
            return {"answer": llm.gerar(state["messages"])}
        except LLMRespostaCortada as cortada:
            # No chat, uma resposta longa cortada no fim ainda é útil
            if ao_gerar:
                ao_gerar(" (...)")
            return {"answer": cortada.parcial + " (...)"}

    return chamar_llm


def _sem_contexto(state: ChatState) -> ChatState:
    return {"answer": state.get("material_warning") or SEM_CONTEXTO}


def _sem_titulo(texto: str, titulo: str | None) -> str:
    """Os trechos levam o título da aula na 1ª linha (ver dividir_em_chunks); na
    fonte o título já aparece, então só o conteúdo vai no excerpt."""
    if titulo and texto.startswith(titulo + "\n"):
        return texto[len(titulo) + 1:]
    return texto


def _formatar_resposta(state: ChatState) -> ChatState:
    if state["answer"].strip().strip('"\'`.') == "SEM_RESPOSTA_NO_MATERIAL":
        return {"answer": state.get("material_warning") or SEM_CONTEXTO, "sources": []}
    fontes: dict[tuple, dict] = {}
    for r in state.get("contexts") or []:
        m = r.metadata
        chave = (m.get("course_id"), m.get("lesson_id"), m.get("version"), r.text)
        if chave not in fontes:
            fontes[chave] = {
                "source_id": m.get("source_id") or hashlib.sha256(repr(chave).encode()).hexdigest(),
                "version": m.get("version"),
                "kind": "retrieved",
                "course_id": m.get("course_id"),
                "module_id": m.get("module_id"),
                "lesson_id": m.get("lesson_id"),
                "lesson_title": m.get("lesson_title"),
                "score": round(r.score, 3),
                # o trecho mais relevante da aula, para quem lê conferir de onde veio a resposta
                "excerpt": _sem_titulo(r.text, m.get("lesson_title")),
            }
    # O número do trecho é interno; se a LLM citá-lo mesmo assim, ele sai do texto.
    resposta = _CITACAO_DE_TRECHO.sub("", state["answer"]).strip()
    if fontes and state.get("material_warning"):
        resposta += "\n\n" + state["material_warning"] + " A resposta considera apenas o material disponível."
    return {"answer": resposta, "sources": list(fontes.values())}


def _tem_contexto(state: ChatState) -> str:
    # Sem trecho relevante não vale gastar a LLM: ela só poderia inventar.
    return "montar_prompt" if state["contexts"] else "sem_contexto"


def criar_grafo_chat(store: VectorStore, embedder: Embedder, llm: LLM, ao_gerar: Callable[[str], None] | None = None):
    grafo = StateGraph(ChatState)
    grafo.add_node("receber_pergunta", _receber_pergunta)
    grafo.add_node("montar_consulta", _montar_consulta)
    grafo.add_node("recuperar_contexto", _criar_recuperador(store, embedder))
    grafo.add_node("montar_prompt", _montar_prompt)
    grafo.add_node("chamar_llm", _criar_chamada_llm(llm, ao_gerar))
    grafo.add_node("sem_contexto", _sem_contexto)
    grafo.add_node("formatar_resposta", _formatar_resposta)

    grafo.add_edge(START, "receber_pergunta")
    grafo.add_edge("receber_pergunta", "montar_consulta")
    grafo.add_edge("montar_consulta", "recuperar_contexto")
    grafo.add_conditional_edges("recuperar_contexto", _tem_contexto, ["montar_prompt", "sem_contexto"])
    grafo.add_edge("montar_prompt", "chamar_llm")
    grafo.add_edge("chamar_llm", "formatar_resposta")
    grafo.add_edge("sem_contexto", "formatar_resposta")
    grafo.add_edge("formatar_resposta", END)
    return grafo.compile()
