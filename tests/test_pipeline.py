"""Limpeza, chunking e índice vetorial."""

import numpy as np

from app.ingestao import Aula, ConteudoVazio, indexar_aula
from app.text_processing import dividir_em_chunks, juntar_linhas, limpar_paginas, limpar_texto, remover_ruido_normativo
from app.vector_store import VectorStore, validar_id

import pytest


def test_limpeza_tira_cabecalho_rodape_e_numero_de_pagina():
    paginas = [
        f"Ministério do Trabalho\nNR-35 Trabalho em altura\nTexto da página {i} que continua\nna linha {i} seguinte.\n{i}"
        for i in range(1, 6)
    ]
    texto = limpar_paginas(paginas)
    assert "Ministério do Trabalho" not in texto
    assert "Texto da página 3 que continua na linha 3 seguinte." in texto
    assert "\n3\n" not in texto


def test_juntar_linhas_separa_itens_e_desfaz_hifenizacao():
    texto = juntar_linhas(["35.1 Objetivo", "35.1.1 Esta Norma esta-", "belece requisitos", "a) planejar;", "b) executar."])
    assert texto.split("\n\n") == [
        "35.1 Objetivo", "35.1.1 Esta Norma estabelece requisitos", "a) planejar;", "b) executar.",
    ]


def test_remove_historico_de_portarias_e_sumario():
    texto = "\n\n".join([
        "NR 35 - TRABALHO EM ALTURA Publicação D.O.U. Portaria SIT nº 313 Alterações/Atualizações D.O.U. Portaria X",
        "SUMÁRIO", "35.1 Objetivo", "35.2 Campo de Aplicação",
        "35.1 Objetivo", "35.1.1 Esta Norma estabelece os requisitos.",
    ])
    assert remover_ruido_normativo(texto).split("\n\n") == [
        "NR 35 - TRABALHO EM ALTURA", "35.1 Objetivo", "35.1.1 Esta Norma estabelece os requisitos.",
    ]


def test_sumario_sem_fim_reconhecivel_nao_apaga_o_texto():
    texto = "SUMÁRIO\n\n1 Introdução\n\nConteúdo importante."
    assert "Conteúdo importante." in remover_ruido_normativo(texto)


def test_limpar_texto_remove_html_do_editor():
    assert limpar_texto("<p>Use <b>capacete</b></p>\r\n\r\n\r\n\r\nsempre") == "Use capacete \n\nsempre"


def test_chunks_levam_titulo_e_metadados():
    chunks = dividir_em_chunks("Frase sobre segurança do trabalho. " * 40, {"lesson_id": "a1"}, titulo="Aula 1")
    assert len(chunks) > 1
    assert all(c.text.startswith("Aula 1\n") for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_busca_traz_a_aula_certa(store, embedder):
    resultado = store.buscar(["curso-a"], embedder.embed_query("quem fornece o EPI ao trabalhador"), k=1)
    assert resultado[0].metadata["lesson_id"] == "nr-06"


def test_busca_nao_vaza_para_curso_de_outra_empresa(store, embedder):
    resultados = store.buscar(["curso-a"], embedder.embed_query("plano estratégico confidencial"), k=10)
    assert all(r.metadata["course_id"] == "curso-a" for r in resultados)


def test_reindexar_aula_substitui_o_conteudo_antigo(store, embedder):
    indexar_aula(Aula(course_id="curso-a", lesson_id="nr-06", title="NR-6", content="Conteúdo novo sobre luvas " * 5),
                 store, embedder)
    textos = [r.text for r in store.curso("curso-a").trechos(lesson_id="nr-06")]
    assert textos and all("luvas" in t for t in textos)
    assert store.curso("curso-a").index.ntotal == len(store.curso("curso-a").chunks)


def test_remover_aula(store):
    assert store.remover_aula("curso-a", "nr-35") > 0
    assert "nr-35" not in store.curso("curso-a").aulas()
    assert store.remover_aula("curso-inexistente", "x") == 0


def test_indice_persiste_em_disco(store, embedder):
    reaberto = VectorStore(embedder.dimension, pasta_base=store.pasta_base)
    assert reaberto.curso("curso-a").aulas() == store.curso("curso-a").aulas()
    assert np.allclose(
        reaberto.curso("curso-a").index.reconstruct(0), store.curso("curso-a").index.reconstruct(0)
    )


def test_aula_sem_texto_e_recusada(store, embedder):
    with pytest.raises(ConteudoVazio):
        indexar_aula(Aula(course_id="curso-a", lesson_id="vazia", title="Vazia", content="   "), store, embedder)


@pytest.mark.parametrize("valor", ["../etc", "a/b", "", "x" * 65, "curso a"])
def test_ids_inseguros_sao_recusados(valor):
    with pytest.raises(ValueError):
        validar_id(valor)
