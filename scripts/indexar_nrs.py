"""Etapas 3 a 5 do pipeline: chunking → embeddings → índice vetorial FAISS.

Monta o curso de demonstração (Normas Regulamentadoras) a partir dos textos
limpos em data/processed/. Cada NR vira uma aula, agrupada nos módulos
definidos em data/curso_nrs.json, exatamente como uma aula enviada pelo Trainify.

Uso (na raiz do projeto):  python -m scripts.indexar_nrs
"""

import json
import shutil
import time

from app import config
from app.embeddings import get_embedder
from app.ingestao import Aula, indexar_aula
from app.vector_store import VectorStore


def main() -> None:
    curso = json.loads((config.DATA_DIR / "curso_nrs.json").read_text(encoding="utf-8"))
    course_id = curso["course_id"]

    pasta = config.INDEX_DIR / course_id
    if pasta.exists():
        shutil.rmtree(pasta)  # reconstrução completa

    print(f"Carregando o modelo de embeddings {config.EMBEDDING_MODEL} (na primeira vez ele é baixado)...")
    embedder = get_embedder()
    store = VectorStore(embedder.dimension)

    inicio = time.time()
    total = 0
    for modulo in curso["modules"]:
        for licao in modulo["lessons"]:
            arquivo = config.PROCESSED_DIR / f"{licao['lesson_id']}.txt"
            if not arquivo.exists():
                print(f"  {licao['lesson_id']}: texto limpo não encontrado, rode scripts.processar_nrs antes")
                continue
            quantidade = indexar_aula(
                Aula(
                    course_id=course_id,
                    tenant_id=config.DEMO_TENANT_ID,
                    module_id=modulo["module_id"],
                    module_title=modulo["title"],
                    lesson_id=licao["lesson_id"],
                    title=licao["title"],
                    content=arquivo.read_text(encoding="utf-8"),
                ),
                store,
                embedder,
            )
            total += quantidade
            print(f"  {licao['lesson_id']}: {quantidade} chunks")

    print(f"\nConcluído! {total} chunks indexados em {pasta} ({time.time() - inicio:.0f}s).")


if __name__ == "__main__":
    main()
