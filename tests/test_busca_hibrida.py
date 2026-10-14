"""Busca híbrida: BM25 (palavras exatas) somado à busca vetorial."""

import numpy as np

from app import config
from app.busca_lexica import IndiceBM25, tokenizar
from app.ingestao import Aula, indexar_aula
from app.vector_store import VectorStore


def test_tokenizacao_mantem_item_da_norma_e_tira_acento():
    assert tokenizar("O que diz o item 35.4.1 da NR-35?") == ["diz", "item", "35.4.1", "nr", "35"]
    assert tokenizar("Proteção, ÁREA") == ["protecao", "area"]


def test_bm25_prefere_o_trecho_com_o_termo_raro():
    indice = IndiceBM25([
        "35.4.1 Todo trabalho em altura deve ser realizado por trabalhador autorizado.",
        "35.5.1 O trabalho em altura exige análise de risco.",
        "Trabalho em altura e trabalho em altura e trabalho.",
    ])
    melhores = indice.buscar("item 35.4.1", k=3)
    assert melhores[0][0] == 0
    assert len(melhores) == 1  # os outros não têm nenhum termo da consulta
    assert indice.buscar("palavra inexistente", k=3) == []


class EmbedderCego:
    """Dá o mesmo vetor para tudo: só o BM25 consegue diferenciar os trechos."""

    dimension = 8

    def _vetor(self):
        v = np.ones(self.dimension, dtype="float32")
        return v / np.linalg.norm(v)

    def embed_passages(self, textos):
        return np.stack([self._vetor() for _ in textos])

    def embed_query(self, texto):
        return self._vetor()


def _store_com_itens(tmp_path):
    embedder = EmbedderCego()
    store = VectorStore(embedder.dimension, pasta_base=tmp_path)
    for numero in range(1, 6):
        indexar_aula(Aula(course_id="c", lesson_id=f"item-{numero}", title=f"Aula {numero}",
                          content=f"35.{numero}.1 Texto de exemplo do item {numero} sobre trabalho em altura e proteção."),
                     store, embedder)
    return store, embedder


def test_hibrida_acha_o_item_exato_que_o_vetor_nao_distingue(tmp_path, monkeypatch):
    store, embedder = _store_com_itens(tmp_path)
    consulta = "o que diz o item 35.4.1"
    vetor = embedder.embed_query(consulta)

    resultados = store.buscar(["c"], vetor, k=2, texto_consulta=consulta)
    assert resultados[0].metadata["lesson_id"] == "item-4"
    # a nota que passa pelo MIN_SCORE continua sendo o cosseno puro
    assert abs(resultados[0].score - 1.0) < 1e-5
    assert resultados[0].relevancia > resultados[0].score

    monkeypatch.setattr(config, "HYBRID_SEARCH", False)
    resultados = store.buscar(["c"], vetor, k=5, texto_consulta=consulta)
    assert all(r.relevancia == r.score for r in resultados)


def test_indice_lexico_acompanha_aula_editada(tmp_path):
    store, embedder = _store_com_itens(tmp_path)
    store.buscar(["c"], embedder.embed_query("x"), k=1, texto_consulta="35.4.1")  # monta o BM25
    indexar_aula(Aula(course_id="c", lesson_id="item-4", title="Aula 4", content="99.9.9 Conteúdo novo da aula editada, agora sobre escadas portáteis."),
                 store, embedder)
    resultados = store.buscar(["c"], embedder.embed_query("x"), k=1, texto_consulta="99.9.9")
    assert resultados[0].metadata["lesson_id"] == "item-4"
