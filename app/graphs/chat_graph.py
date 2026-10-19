"""Grafo de RAG do chat, orquestrado com LangGraph.

    receber_pergunta → montar_consulta ─┬─ (síntese, com aula aberta) → carregar_aula ─────┐
                                        └─ (demais perguntas) ───────→ recuperar_contexto ─┤
                                                   montar_prompt → chamar_llm → formatar_resposta

Só o nó chamar_llm usa a LLM externa. Perguntas de seguimento ("e para altura
acima de 2 m?") são resolvidas sem LLM: a consulta de busca junta a pergunta
atual com a última pergunta do usuário quando há indicação de continuidade.

Com a aula aberta (lesson_id), só o material daquela aula é usado: as outras aulas
não concorrem na busca. Perguntas de síntese ("resuma", "pontos principais") não
passam pela busca: recebem a aula inteira (ou condensada, sem LLM, se for longa).
"""

import re
import hashlib
from collections.abc import Callable
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app import config
from app.aula_inteira import condensar, e_pergunta_de_sintese, reconstruir_texto
from app.embeddings import Embedder
from app.graphs.quiz_graph import MaterialInsuficiente
from app.llm import LLM, LLMRespostaCortada
from app.prompts import AULA_INTEIRA_INSTRUCAO, sistema_para
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
    lesson_title: str          # título da aula aberta, quando houver
    whole_lesson: bool         # a aula inteira foi enviada (pergunta de síntese)
    other_lessons: list[str]   # aulas onde o assunto aparece, quando não está na aula aberta
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

SEM_MATERIAL_NA_AULA = (
    "Esta aula ainda não tem material de texto para o assistente consultar "
    "(texto, transcrição ou PDF). Se o material acabou de ser enviado, tente de novo em alguns minutos."
)
SEM_MATERIAL_NOS_CURSOS = (
    "Ainda não há material de texto disponível para o assistente nos seus cursos. "
    "Se o material acabou de ser enviado, tente de novo em alguns minutos."
)
MATERIAL_SENDO_ATUALIZADO = "O material está sendo atualizado. Tente novamente em instantes."


def _fora_da_aula(titulo: str | None, outras: list[str]) -> str:
    """Resposta quando o assunto não está na aula aberta. Diz onde ele está, sem
    apresentar o conteúdo de outra aula como se fosse desta."""
    aula = f' ("{titulo}")' if titulo else ""
    if outras:
        lista = "; ".join(f'"{t}"' for t in outras)
        return (f"Esse assunto não está nesta aula{aula}. Ele aparece em: {lista}. "
                "Abra essa aula para perguntar sobre ele.")
    return (f"Não encontrei esse assunto nesta aula{aula}. Tente perguntar de outro jeito "
            "ou com termos que aparecem na aula. Se a dúvida continuar, fale com o instrutor.")


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


def _escolher_caminho(state: ChatState) -> str:
    if state.get("lesson_id") and e_pergunta_de_sintese(state["question"]):
        return "carregar_aula"
    return "recuperar_contexto"


def _exigir_material_da_aula(store: VectorStore, course_ids: list[str], lesson_id: str) -> str | None:
    """Curso onde a aula está. Aula sem material pronto: atualização em andamento
    vira aviso (None); aula sem texto nenhum vira 422, com a explicação."""
    curso = store.curso_da_aula(course_ids, lesson_id)
    if curso and store.tem_material([curso], lesson_id):
        return curso
    if curso and store.situacao(curso)["lessons"].get(lesson_id, {}).get("status") == "PROCESSING":
        return None
    raise MaterialInsuficiente(SEM_MATERIAL_NA_AULA)


def _criar_carregador_de_aula(store: VectorStore):
    def carregar_aula(state: ChatState) -> ChatState:
        aula = state["lesson_id"]
        curso = _exigir_material_da_aula(store, state.get("course_ids") or [], aula)
        if curso is None:
            return {"contexts": [], "material_warning": MATERIAL_SENDO_ATUALIZADO, "whole_lesson": True}
        trechos = store.curso(curso).trechos(lesson_id=aula)
        metadata = dict(trechos[0].metadata)
        titulo = metadata.get("lesson_title")
        texto, condensado = condensar(reconstruir_texto([t.text for t in trechos], titulo))
        metadata["condensed"] = condensado
        return {
            "contexts": [Resultado(texto, metadata, 1.0)],
            "material_warning": "",
            "lesson_title": titulo or "",
            "whole_lesson": True,
        }

    return carregar_aula


def _criar_recuperador(store: VectorStore, embedder: Embedder):
    def recuperar_contexto(state: ChatState) -> ChatState:
        consulta = state["search_query"]
        course_ids = state.get("course_ids") or []
        aula = state.get("lesson_id")
        vetor = embedder.embed_query(consulta)

        if aula:
            # Aula aberta: só ela entra na busca. As outras aulas não concorrem.
            curso = _exigir_material_da_aula(store, course_ids, aula)
            if curso is None:
                return {"contexts": [], "material_warning": MATERIAL_SENDO_ATUALIZADO}
            candidatos = store.buscar([curso], vetor, k=config.TOP_K * 2, texto_consulta=consulta, lesson_id=aula)
            relevantes = [r for r in candidatos if r.score >= config.MIN_SCORE]
            titulo = next((c.metadata.get("lesson_title") for c in candidatos if c.metadata.get("lesson_title")), "")
            outras: list[str] = []
            if not relevantes:
                # Só para dizer onde o assunto está; o conteúdo de outra aula não vai para a resposta
                for r in store.buscar(course_ids, vetor, k=config.TOP_K * 2, texto_consulta=consulta):
                    nome = r.metadata.get("lesson_title")
                    if r.score >= config.MIN_SCORE and r.metadata.get("lesson_id") != aula and nome and nome not in outras:
                        outras.append(nome)
            return {"contexts": relevantes[:config.TOP_K], "material_warning": "",
                    "lesson_title": titulo, "other_lessons": outras[:2]}

        if not store.tem_material(course_ids):
            processando = any(
                e.get("status") == "PROCESSING"
                for c in course_ids for e in store.situacao(c)["lessons"].values()
            )
            if processando:
                return {"contexts": [], "material_warning": MATERIAL_SENDO_ATUALIZADO}
            raise MaterialInsuficiente(SEM_MATERIAL_NOS_CURSOS)

        # Busca híbrida: vetor (sentido) + BM25 (palavras exatas, como "item 35.4.1")
        candidatos = store.buscar(course_ids, vetor, k=config.TOP_K * 2, texto_consulta=consulta)
        relevantes = [r for r in candidatos if r.score >= config.MIN_SCORE]

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
    sistema = sistema_para(state["role"])
    if state.get("whole_lesson"):
        sistema += "\n" + AULA_INTEIRA_INSTRUCAO
        if state["contexts"] and state["contexts"][0].metadata.get("condensed"):
            sistema += ("\n- A aula é longa: abaixo vai uma versão condensada, com os títulos das seções "
                        "e a primeira frase de cada item, cobrindo a aula do começo ao fim.")
    sistema += f"\n\nTRECHOS DO MATERIAL:\n{trechos}"

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


def _resposta_sem_material(state: ChatState) -> str:
    if state.get("material_warning"):
        return state["material_warning"]
    if state.get("lesson_id"):
        return _fora_da_aula(state.get("lesson_title"), state.get("other_lessons") or [])
    return SEM_CONTEXTO


def _sem_contexto(state: ChatState) -> ChatState:
    return {"answer": _resposta_sem_material(state)}


def _sem_titulo(texto: str, titulo: str | None) -> str:
    """Os trechos levam o título da aula na 1ª linha (ver dividir_em_chunks); na
    fonte o título já aparece, então só o conteúdo vai no excerpt."""
    if titulo and texto.startswith(titulo + "\n"):
        return texto[len(titulo) + 1:]
    return texto


def _formatar_resposta(state: ChatState) -> ChatState:
    if state["answer"].strip().strip('"\'`.') == "SEM_RESPOSTA_NO_MATERIAL":
        return {"answer": _resposta_sem_material(state), "sources": []}
    fontes: dict[tuple, dict] = {}
    for r in state.get("contexts") or []:
        m = r.metadata
        aula_completa = bool(state.get("whole_lesson"))
        tipo = ("lesson_condensed" if m.get("condensed") else "lesson") if aula_completa else "retrieved"
        chave = (m.get("course_id"), m.get("lesson_id"), m.get("version"), r.text)
        identificador = hashlib.sha256(repr((tipo, chave)).encode()).hexdigest() if aula_completa else (
            m.get("source_id") or hashlib.sha256(repr(chave).encode()).hexdigest()
        )
        if chave not in fontes:
            fontes[chave] = {
                "source_id": identificador,
                "version": m.get("version"),
                "kind": tipo,
                "course_id": m.get("course_id"),
                "module_id": m.get("module_id"),
                "lesson_id": m.get("lesson_id"),
                "lesson_title": m.get("lesson_title"),
                "score": None if aula_completa else round(r.score, 3),
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
    grafo.add_node("carregar_aula", _criar_carregador_de_aula(store))
    grafo.add_node("recuperar_contexto", _criar_recuperador(store, embedder))
    grafo.add_node("montar_prompt", _montar_prompt)
    grafo.add_node("chamar_llm", _criar_chamada_llm(llm, ao_gerar))
    grafo.add_node("sem_contexto", _sem_contexto)
    grafo.add_node("formatar_resposta", _formatar_resposta)

    grafo.add_edge(START, "receber_pergunta")
    grafo.add_edge("receber_pergunta", "montar_consulta")
    grafo.add_conditional_edges("montar_consulta", _escolher_caminho, ["carregar_aula", "recuperar_contexto"])
    grafo.add_conditional_edges("carregar_aula", _tem_contexto, ["montar_prompt", "sem_contexto"])
    grafo.add_conditional_edges("recuperar_contexto", _tem_contexto, ["montar_prompt", "sem_contexto"])
    grafo.add_edge("montar_prompt", "chamar_llm")
    grafo.add_edge("chamar_llm", "formatar_resposta")
    grafo.add_edge("sem_contexto", "formatar_resposta")
    grafo.add_edge("formatar_resposta", END)
    return grafo.compile()
