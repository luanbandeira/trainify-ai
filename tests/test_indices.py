import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from app.ingestao import Aula, indexar_aula
from app.vector_store import VectorStore


def test_falha_ao_salvar_preserva_indice_em_memoria_e_disco(store, embedder, monkeypatch):
    anterior = store.curso("curso-a").trechos()
    replace = Path.replace

    def falhar_snapshot(path, destino):
        if path.name == "snapshot.npz.tmp":
            raise OSError("Disco indisponível")
        return replace(path, destino)

    monkeypatch.setattr(Path, "replace", falhar_snapshot)
    with pytest.raises(OSError):
        indexar_aula(Aula(course_id="curso-a", lesson_id="nr-06", title="Nova", content="Luvas novas " * 20), store, embedder)
    assert store.curso("curso-a").trechos() == anterior
    assert VectorStore(embedder.dimension, store.pasta_base).curso("curso-a").trechos() == anterior


def test_leitor_mantem_snapshot_consistente_durante_atualizacao(store, embedder):
    anterior = store.curso("curso-a")
    trechos = anterior.trechos()
    indexar_aula(Aula(course_id="curso-a", lesson_id="nr-06", title="Nova", content="Luvas novas " * 20), store, embedder)
    assert anterior.trechos() == trechos
    assert store.curso("curso-a").trechos() != trechos


def test_indice_legado_continua_legivel_e_migra_na_escrita(tmp_path, embedder):
    pasta = tmp_path / "antigo"
    pasta.mkdir()
    indice = faiss.IndexFlatIP(embedder.dimension)
    indice.add(np.ones((1, embedder.dimension), dtype="float32"))
    faiss.write_index(indice, str(pasta / "index.faiss"))
    (pasta / "chunks.jsonl").write_text(json.dumps({"text": "antigo", "metadata": {"lesson_id": "a"}}), encoding="utf-8")
    store = VectorStore(embedder.dimension, tmp_path)
    assert store.curso("antigo").aulas() == {"a": 1}
    store.remover_aula("antigo", "a")
    assert (pasta / "snapshot.npz").exists()
    assert VectorStore(embedder.dimension, tmp_path).curso("antigo").aulas() == {}
