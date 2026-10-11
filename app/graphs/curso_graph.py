"""Grafo que propõe um curso inteiro a partir dos documentos da empresa.

    ler_documentos → indexar_material → mapear_material → propor_estrutura → validar_estrutura
                                                                ↑                     │
                                                                └── nova tentativa ───┘ (JSON inválido)
                                                                                      ↓
                                                                                escrever_aulas

A estrutura sai de UMA chamada à LLM, a partir de um "mapa" montado sem LLM
(amostras espalhadas do material com a página de origem). Depois, cada aula é
escrita só com os trechos recuperados para ela, como no chat. Assim o curso
cabe no limite gratuito da Groq, e cada aula sabe de quais páginas saiu.

Nada é salvo no Trainify aqui: o instrutor revisa e aprova a proposta.
"""

import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app import config
from app.embeddings import Embedder
from app.llm import LLM, LLMLimiteAtingido, LLMRespostaVazia
from app.prompts import AULA_SISTEMA, ESTRUTURA_SISTEMA
from app.text_processing import Chunk, extrair_texto_pdf, limpar_texto, paginas_limpas, trechos_por_pagina
from app.vector_store import CourseIndex

Progresso = Callable[[int, str], None]


class MaterialInsuficiente(ValueError):
    """Nenhum documento tinha texto aproveitável."""


class EstruturaInvalida(RuntimeError):
    """A LLM não conseguiu propor uma estrutura válida depois de todas as tentativas."""


class Documento(TypedDict, total=False):
    name: str
    pdf: bytes
    text: str


class CursoState(TypedDict, total=False):
    # entrada
    documents: list[Documento]
    min_modules: int
    max_modules: int
    instructions: str | None
    # intermediários
    chunks: list[Chunk]
    warnings: list[str]
    material_map: str
    messages: list[BaseMessage]
    raw_output: str
    errors: list[str]
    attempts: int
    structure: dict
    # saída
    proposal: dict


def gerar_com_paciencia(llm: LLM, mensagens: list[BaseMessage], aviso: Progresso | None = None) -> str:
    """Um curso faz dezenas de chamadas; uma falha passageira não pode derrubar
    tudo. No limite por minuto da Groq, espera e tenta de novo; numa resposta
    vazia (acontece às vezes com modelos que "pensam" antes), tenta de novo na hora."""
    vazias = 0
    for tentativa in range(config.CURSO_TENTATIVAS_LIMITE):
        try:
            return llm.gerar(mensagens)
        except LLMRespostaVazia:
            vazias += 1
            if vazias >= 3:
                raise
            continue
        except LLMLimiteAtingido:
            if tentativa == config.CURSO_TENTATIVAS_LIMITE - 1:
                raise
            if aviso:
                aviso(-1, "Aguardando o limite de uso da IA liberar")
            time.sleep(config.CURSO_ESPERA_LIMITE_S)
    raise AssertionError("inalcançável")


def _ler_documentos(state: CursoState) -> CursoState:
    trechos: list[Chunk] = []
    avisos: list[str] = []
    for doc in state["documents"]:
        nome = doc["name"]
        if doc.get("pdf"):
            paginas = paginas_limpas(extrair_texto_pdf(doc["pdf"]))
        else:
            paginas = [limpar_texto(doc.get("text") or "")]
        do_documento = trechos_por_pagina(paginas, {"document": nome, "lesson_id": nome}, titulo=None)
        if not do_documento:
            avisos.append(f'"{nome}" não tem texto aproveitável (pode ser um PDF escaneado, que é uma imagem) e foi ignorado.')
        trechos.extend(do_documento)
    if not trechos:
        raise MaterialInsuficiente(
            "Nenhum dos documentos tem texto aproveitável. Envie PDFs com texto selecionável (não escaneados)."
        )
    return {"chunks": trechos, "warnings": avisos}


def _criar_indexador(embedder: Embedder, progresso: Progresso, compartilhado: dict):
    def indexar_material(state: CursoState) -> CursoState:
        progresso(10, "Lendo e organizando o material")
        # Índice só em memória: dura o tempo da geração e não ocupa o disco
        indice = CourseIndex("rascunho", Path(tempfile.gettempdir()) / "trainify-rascunho-inexistente", embedder.dimension)
        por_documento: dict[str, list[Chunk]] = {}
        for c in state["chunks"]:
            por_documento.setdefault(c.metadata["document"], []).append(c)
        for nome, trechos in por_documento.items():
            indice.substituir_aula(nome, trechos, embedder.embed_passages([t.text for t in trechos]))
        # O índice não é serializável no estado do grafo; fica num objeto que só
        # este grafo (montado a cada geração) enxerga
        compartilhado["indice"] = indice
        return {"warnings": state.get("warnings", [])}

    return indexar_material


def _mapear_material(state: CursoState) -> CursoState:
    """Amostras espalhadas de cada documento, com a página. Sem LLM."""
    por_documento: dict[str, list[Chunk]] = {}
    for c in state["chunks"]:
        por_documento.setdefault(c.metadata["document"], []).append(c)

    orcamento = config.CURSO_MAPA_CARACTERES
    por_doc = max(4, orcamento // (len(por_documento) * config.CURSO_MAPA_AMOSTRA))
    partes = []
    for nome, trechos in por_documento.items():
        passo = max(1, len(trechos) / por_doc)
        escolhidos = [trechos[int(i * passo)] for i in range(min(por_doc, len(trechos)))]
        linhas = [f"[p. {c.metadata['page']}] {c.text[:config.CURSO_MAPA_AMOSTRA].strip()}" for c in escolhidos]
        partes.append(f"=== Documento: {nome} ({len(trechos)} trechos)\n" + "\n".join(linhas))
    return {"material_map": "\n\n".join(partes), "attempts": 0, "errors": []}


def _criar_montador_estrutura(progresso: Progresso):
    def montar_prompt_estrutura(state: CursoState) -> CursoState:
        progresso(20, "Propondo a estrutura do curso")
        pedido = "MAPA DO MATERIAL:\n" + state["material_map"]
        if state.get("instructions"):
            pedido = f"Orientação do instrutor: {state['instructions']}\n\n" + pedido
        mensagens: list[BaseMessage] = [
            SystemMessage(content=ESTRUTURA_SISTEMA.format(min_modulos=state["min_modules"], max_modulos=state["max_modules"])),
            HumanMessage(content=pedido),
        ]
        if state.get("errors"):
            mensagens.append(HumanMessage(content=(
                "Sua resposta anterior foi recusada pelos seguintes problemas:\n- " + "\n- ".join(state["errors"])
                + "\nGere o JSON completo de novo, corrigindo todos eles."
            )))
        return {"messages": mensagens}

    return montar_prompt_estrutura


def _criar_chamada_estrutura(llm: LLM, progresso: Progresso):
    def propor_estrutura(state: CursoState) -> CursoState:
        return {"raw_output": gerar_com_paciencia(llm, state["messages"], progresso), "attempts": state["attempts"] + 1}

    return propor_estrutura


def validar_estrutura(texto: str, documentos: list[str], min_modulos: int, max_modulos: int) -> tuple[dict | None, list[str]]:
    try:
        inicio, fim = texto.find("{"), texto.rfind("}")
        dados = json.loads(texto[inicio: fim + 1]) if inicio >= 0 and fim > inicio else None
    except json.JSONDecodeError as erro:
        return None, [f"JSON inválido: {erro}"]
    if not isinstance(dados, dict):
        return None, ["a resposta não contém um objeto JSON"]

    erros: list[str] = []
    titulo = str(dados.get("title") or "").strip()
    if not titulo:
        erros.append('falta o "title" do curso')
    modulos = dados.get("modules")
    if not isinstance(modulos, list) or not (min_modulos <= len(modulos) <= max_modulos):
        erros.append(f"o curso precisa ter entre {min_modulos} e {max_modulos} módulos")
        modulos = modulos if isinstance(modulos, list) else []

    conhecidos = {d.lower(): d for d in documentos}
    normalizados = []
    for n, m in enumerate(modulos, start=1):
        if not isinstance(m, dict) or not str(m.get("title") or "").strip():
            erros.append(f"módulo {n} sem título")
            continue
        aulas = m.get("lessons") if isinstance(m.get("lessons"), list) else []
        if not aulas:
            erros.append(f"módulo {n} sem aulas")
        aulas_ok = []
        for a in aulas:
            if not isinstance(a, dict) or not str(a.get("title") or "").strip():
                erros.append(f"módulo {n} tem aula sem título")
                continue
            topicos = [str(t).strip() for t in (a.get("topics") or []) if str(t).strip()]
            docs = [conhecidos[str(d).lower()] for d in (a.get("documents") or []) if str(d).lower() in conhecidos]
            aulas_ok.append({"title": str(a["title"]).strip(), "topics": topicos[:6], "documents": docs})
        normalizados.append({
            "title": str(m["title"]).strip(),
            "objective": str(m.get("objective") or "").strip(),
            "lessons": aulas_ok,
        })

    if erros:
        return None, erros
    return {
        "title": titulo,
        "description": str(dados.get("description") or "").strip(),
        "modules": normalizados,
    }, []


def _validar(state: CursoState) -> CursoState:
    documentos = sorted({c.metadata["document"] for c in state["chunks"]})
    estrutura, erros = validar_estrutura(state["raw_output"], documentos, state["min_modules"], state["max_modules"])
    if estrutura:
        return {"structure": estrutura, "errors": []}
    return {"errors": erros}


def _proximo_passo(state: CursoState) -> str:
    if state.get("structure"):
        return "escrever"
    if state["attempts"] >= config.QUIZ_MAX_ATTEMPTS:
        return "falhou"
    return "tentar_de_novo"


def _falhou(state: CursoState) -> CursoState:
    raise EstruturaInvalida("A IA não conseguiu propor a estrutura do curso: " + "; ".join(state["errors"]))


def _paginas(trechos) -> str:
    """'4-7, 12' a partir das páginas dos trechos usados."""
    numeros = sorted({t.metadata.get("page") for t in trechos if t.metadata.get("page")})
    faixas, inicio, anterior = [], None, None
    for n in numeros:
        if inicio is None:
            inicio = anterior = n
        elif n == anterior + 1:
            anterior = n
        else:
            faixas.append(f"{inicio}-{anterior}" if inicio != anterior else str(inicio))
            inicio = anterior = n
    if inicio is not None:
        faixas.append(f"{inicio}-{anterior}" if inicio != anterior else str(inicio))
    return ", ".join(faixas)


def _criar_escritor(llm: LLM, embedder: Embedder, progresso: Progresso, compartilhado: dict):
    def escrever_aulas(state: CursoState) -> CursoState:
        indice: CourseIndex = compartilhado.pop("indice")
        estrutura = state["structure"]
        total = sum(len(m["lessons"]) for m in estrutura["modules"])
        feitas = 0
        modulos = []
        for m in estrutura["modules"]:
            aulas = []
            for a in m["lessons"]:
                feitas += 1
                progresso(25 + int(70 * (feitas - 1) / total), f"Escrevendo a aula {feitas} de {total}: {a['title']}")

                consulta = a["title"] + ". " + "; ".join(a["topics"])
                candidatos = indice.buscar(embedder.embed_query(consulta), k=config.CURSO_TRECHOS_POR_AULA * 3)
                if a["documents"]:
                    preferidos = [r for r in candidatos if r.metadata["document"] in a["documents"]]
                    candidatos = preferidos or candidatos
                trechos = candidatos[: config.CURSO_TRECHOS_POR_AULA]

                material = "\n\n".join(f"--- {r.text}" for r in trechos)
                pedido = (f"Aula: {a['title']}\nTópicos: " + "; ".join(a["topics"])
                          + f"\n\nTRECHOS DO MATERIAL:\n{material}")
                conteudo = gerar_com_paciencia(llm, [SystemMessage(content=AULA_SISTEMA), HumanMessage(content=pedido)], progresso)

                fontes: dict[str, list] = {}
                for r in trechos:
                    fontes.setdefault(r.metadata["document"], []).append(r)
                aulas.append({
                    "title": a["title"],
                    "topics": a["topics"],
                    "content": conteudo.strip(),
                    "sources": [{"document": doc, "pages": _paginas(ts)} for doc, ts in fontes.items()],
                })
            modulos.append({"title": m["title"], "objective": m["objective"], "lessons": aulas})
        return {"proposal": {**estrutura, "modules": modulos, "warnings": state.get("warnings", [])}}

    return escrever_aulas


def criar_grafo_curso(embedder: Embedder, llm_estrutura: LLM, llm_aulas: LLM, progresso: Progresso):
    compartilhado: dict = {}

    grafo = StateGraph(CursoState)
    grafo.add_node("ler_documentos", _ler_documentos)
    grafo.add_node("indexar_material", _criar_indexador(embedder, progresso, compartilhado))
    grafo.add_node("mapear_material", _mapear_material)
    grafo.add_node("montar_prompt_estrutura", _criar_montador_estrutura(progresso))
    grafo.add_node("propor_estrutura", _criar_chamada_estrutura(llm_estrutura, progresso))
    grafo.add_node("validar_estrutura", _validar)
    grafo.add_node("falhou", _falhou)
    grafo.add_node("escrever_aulas", _criar_escritor(llm_aulas, embedder, progresso, compartilhado))

    grafo.add_edge(START, "ler_documentos")
    grafo.add_edge("ler_documentos", "indexar_material")
    grafo.add_edge("indexar_material", "mapear_material")
    grafo.add_edge("mapear_material", "montar_prompt_estrutura")
    grafo.add_edge("montar_prompt_estrutura", "propor_estrutura")
    grafo.add_edge("propor_estrutura", "validar_estrutura")
    grafo.add_conditional_edges(
        "validar_estrutura", _proximo_passo,
        {"escrever": "escrever_aulas", "tentar_de_novo": "montar_prompt_estrutura", "falhou": "falhou"},
    )
    grafo.add_edge("escrever_aulas", END)
    grafo.add_edge("falhou", END)
    return grafo.compile()
