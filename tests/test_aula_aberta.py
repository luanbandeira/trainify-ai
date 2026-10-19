"""Cenários da integração com o Trainify (aula aberta, resumo, cursos sem material)."""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.aula_inteira import condensar, e_pergunta_de_sintese, reconstruir_texto
from app.graphs.chat_graph import criar_grafo_chat
from app.graphs.quiz_graph import MaterialInsuficiente
from app.ingestao import Aula, indexar_aula
from tests.dubles import LLMFalsa

CIPA = (
    "5.1 Objetivo\n\n5.1.1 Esta norma estabelece os parâmetros da Comissão Interna de Prevenção de Acidentes e de "
    "Assédio, a CIPA.\n\n5.4 Constituição\n\n5.4.6 O mandato dos membros eleitos da CIPA terá a duração de um ano, "
    "permitida uma reeleição.\n\n5.7 Treinamento\n\n5.7.4.3 O treinamento pode ser realizado na modalidade de "
    "ensino a distância ou semipresencial. "
)
GLOSSARIO = "Glossário. Projeto pedagógico: documento que registra o objetivo da aprendizagem. Log: registro informatizado de acesso. " * 4


@pytest.fixture
def cursos(store, embedder):
    indexar_aula(Aula(course_id="curso-a", lesson_id="nr-05", module_id="m1", title="NR-5: CIPA", content=CIPA * 2),
                 store, embedder)
    indexar_aula(Aula(course_id="curso-a", lesson_id="nr-01", module_id="m1", title="NR-1: Disposições gerais",
                      content=GLOSSARIO), store, embedder)
    return store


def _chat(store, embedder, llm, **entrada):
    entrada.setdefault("course_ids", ["curso-a"])
    return criar_grafo_chat(store, embedder, llm).invoke(entrada)


# ── A: resumo da aula aberta ──

def test_resumo_da_aula_usa_a_aula_inteira_e_so_ela(cursos, embedder):
    llm = LLMFalsa("A CIPA tem mandato de um ano.")
    saida = _chat(cursos, embedder, llm, question="quais os pontos mais importantes dessa aula", lesson_id="nr-05")

    sistema = llm.chamadas[0][0].content
    assert "AULA INTEIRA" in sistema
    assert "mandato dos membros eleitos" in sistema and "5.7.4.3" in sistema   # a aula toda, não trechos soltos
    assert "Projeto pedagógico" not in sistema                                 # nada da outra aula
    assert [f["lesson_id"] for f in saida["sources"]] == ["nr-05"]


# ── B: pergunta pontual dentro da aula ──

def test_pergunta_pontual_busca_so_na_aula_aberta(cursos, embedder):
    llm = LLMFalsa("Um ano, permitida uma reeleição.")
    saida = _chat(cursos, embedder, llm, question="qual a duração do mandato dos membros da CIPA?", lesson_id="nr-05")
    assert {f["lesson_id"] for f in saida["sources"]} == {"nr-05"}
    assert "Projeto pedagógico" not in llm.chamadas[0][0].content


# ── C: assunto de outra aula ──

def test_assunto_de_outra_aula_e_apontado_sem_trazer_o_conteudo(cursos, embedder):
    llm = LLMFalsa()
    saida = _chat(cursos, embedder, llm, question="o que é projeto pedagógico e registro informatizado de acesso",
                  lesson_id="nr-05")
    assert not llm.chamadas                                   # nada da aula aberta: a LLM nem é chamada
    assert 'não está nesta aula ("NR-5: CIPA")' in saida["answer"]
    assert '"NR-1: Disposições gerais"' in saida["answer"]
    assert saida["sources"] == []


def test_recusa_da_llm_dentro_da_aula_tambem_aponta_a_aula_certa(cursos, embedder):
    saida = _chat(cursos, embedder, LLMFalsa("SEM_RESPOSTA_NO_MATERIAL"),
                  question="qual o mandato da CIPA", lesson_id="nr-05")
    assert "nesta aula" in saida["answer"] and saida["sources"] == []


# ── D e E: sem aula aberta, e isolamento ──

def test_sem_aula_aberta_busca_em_todos_os_cursos_permitidos(cursos, embedder):
    saida = _chat(cursos, embedder, LLMFalsa("ok"), question="projeto pedagógico registro informatizado")
    assert "nr-01" in {f["lesson_id"] for f in saida["sources"]}


def test_aula_de_curso_fora_da_lista_nunca_e_usada(cursos, embedder):
    # a aula existe, mas no curso-b, que não está em course_ids
    with pytest.raises(MaterialInsuficiente):
        _chat(cursos, embedder, LLMFalsa(), question="plano estratégico", lesson_id="segredo")


# ── G: sem material ──

def test_aula_sem_material_e_422_explicado(cursos, embedder):
    with pytest.raises(MaterialInsuficiente, match="Esta aula ainda não tem material"):
        _chat(cursos, embedder, LLMFalsa(), question="resuma a aula", lesson_id="aula-so-video")


def test_api_chat_sem_material_responde_422_com_detail(cursos, embedder):
    main.app.dependency_overrides = {
        main.obter_store: lambda: cursos, main.obter_embedder: lambda: embedder, main.obter_llm: lambda: LLMFalsa(),
    }
    try:
        resposta = TestClient(main.app).post("/api/chat", json={"question": "Do que trata?", "course_ids": ["curso-vazio"]})
    finally:
        main.app.dependency_overrides = {}
    assert resposta.status_code == 422
    assert "Ainda não há material" in resposta.json()["detail"]


# ── F: /api/courses só com conteúdo de fato ──

def test_curso_registrado_mas_vazio_fica_fora_da_lista(cursos):
    cursos.remover_aula("curso-b", "segredo")                 # o índice continua existindo, sem trechos
    assert cursos.existe("curso-b")
    assert "curso-b" not in cursos.cursos_indexados()
    assert "curso-a" in cursos.cursos_indexados()


# ── peças sem LLM ──

@pytest.mark.parametrize("pergunta, sintese", [
    ("quais os pontos mais importantes dessa aula", True), ("resuma a aula", True), ("do que trata essa aula?", True),
    ("qual a duração do mandato dos membros da CIPA?", False), ("o que é um inventário de riscos?", False),
])
def test_deteccao_de_pergunta_de_sintese(pergunta, sintese):
    assert e_pergunta_de_sintese(pergunta) is sintese


def test_reconstrucao_desfaz_a_sobreposicao_dos_trechos(cursos):
    trechos = cursos.curso("curso-a").trechos(lesson_id="nr-05")
    texto = reconstruir_texto([t.text for t in trechos], "NR-5: CIPA")
    assert texto.count("5.4.6 O mandato") == 2                # o conteúdo tem a seção duas vezes (CIPA * 2)
    assert "NR-5: CIPA\n" not in texto


def test_aula_longa_e_condensada_cobrindo_do_comeco_ao_fim():
    secoes = "\n\n".join(f"{n}.1 Seção {n}\n\n{n}.1.1 Primeira frase da seção {n}. Detalhe longo. " + "x " * 300
                         for n in range(1, 30))
    texto, condensado = condensar(secoes, limite=3000)
    assert condensado and len(texto) <= 3000
    assert "1.1 Seção 1" in texto and "29.1 Seção 29" in texto   # títulos do começo ao fim
    assert "x x x x x" not in texto                              # o miolo dos itens fica de fora


def test_titulos_acima_do_limite_preservam_inicio_meio_e_fim():
    titulos = [f"{n:03}.1 Seção {n:03} " + "conteúdo " * 8 for n in range(1, 101)]
    texto, reduzido = condensar("\n\n".join(titulos), limite=600)
    assert reduzido and len(texto) <= 600
    partes = texto.split("\n\n")
    numeros = [int(p.split(".")[0]) for p in partes]
    assert numeros[0] == 1 and numeros[-1] == 100
    assert numeros == sorted(set(numeros))
    assert any(30 <= numero <= 70 for numero in numeros)
    assert all(parte in {titulo.strip() for titulo in titulos} for parte in partes)


def test_limite_pequeno_distribui_espaco_entre_extremos():
    texto, reduzido = condensar("1.1 Introdução extensa à segurança\n\n2.1 Conclusão extensa da segurança", limite=30)
    assert reduzido and len(texto) <= 30
    assert texto.startswith("1.1") and "\n\n2.1" in texto


def test_aula_que_cabe_nao_e_modificada():
    original = "1.1 Introdução\n\nConteúdo da aula.\n\n2.1 Conclusão"
    assert condensar(original, limite=len(original)) == (original, False)


@pytest.mark.parametrize("limite", [0, -1])
def test_limite_de_condensacao_invalido(limite):
    with pytest.raises(ValueError, match="positivo"):
        condensar("Material", limite=limite)
