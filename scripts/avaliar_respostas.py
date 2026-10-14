"""Gera respostas para revisão humana, usando materiais fictícios e índice temporário."""

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from app import config
from app.graphs.chat_graph import SEM_CONTEXTO, criar_grafo_chat
from app.ingestao import Aula, indexar_aula
from app.vector_store import VectorStore

CRITERIOS = ("correta", "fundamentada", "completa", "fontes_sustentam", "perfil_adequado")
BASE = config.DATA_DIR / "avaliacao" / "respostas_corporativas.json"


def executar(base, embedder, llm):
    casos = []
    with tempfile.TemporaryDirectory(prefix="trainify-avaliacao-") as pasta:
        store = VectorStore(embedder.dimension, Path(pasta))
        for material in base["materials"]:
            indexar_aula(Aula(**material), store, embedder)
        grafo = criar_grafo_chat(store, embedder, llm)
        for caso in base["cases"]:
            registro = {**caso, "review": {c: None for c in CRITERIOS}, "reviewer": "", "notes": ""}
            try:
                resposta = grafo.invoke({
                    "question": caso["question"], "course_ids": caso["course_ids"],
                    "role": caso.get("role", "STUDENT"), "history": caso.get("history", []),
                })
                registro.update(answer=resposta["answer"], sources=resposta["sources"], error=None)
                registro["checks"] = {
                    "refusal_matches": (resposta["answer"] == SEM_CONTEXTO) == caso["expect_refusal"],
                    "sources_in_scope": all(s["course_id"] in caso["course_ids"] for s in resposta["sources"]),
                    "expected_lessons_found": all(
                        lid in {s["lesson_id"] for s in resposta["sources"]} for lid in caso["expected_lessons"]
                    ),
                }
            except Exception as erro:
                registro.update(answer=None, sources=[], error=type(erro).__name__, checks={})
            casos.append(registro)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset_version": base["version"],
        "configuration": {"model": config.LLM_MODEL, "embedding": config.EMBEDDING_MODEL,
                          "min_score": config.MIN_SCORE, "top_k": config.TOP_K, "hybrid": config.HYBRID_SEARCH},
        "cases": casos,
    }


def resumir(relatorio):
    casos = relatorio["cases"]
    revisados = [c for c in casos if not c["error"] and c.get("reviewer", "").strip()
                 and all(type(c.get("review", {}).get(k)) is bool for k in CRITERIOS)]
    return {
        "total": len(casos), "falhas_execucao": sum(bool(c["error"]) for c in casos),
        "revisados": len(revisados), "pendentes": len(casos) - len(revisados),
        "criterios": {k: sum(c["review"][k] for c in revisados) / len(revisados) if revisados else None
                      for k in CRITERIOS},
        "aprovacao_integral": sum(all(c["review"].values()) for c in revisados) / len(revisados) if revisados else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--gerar", action="store_true", help="usa a LLM configurada e consome a cota do provedor")
    modo.add_argument("--resumir", type=Path, help="resume somente revisões humanas preenchidas")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--saida", type=Path)
    args = parser.parse_args()
    if args.resumir:
        print(json.dumps(resumir(json.loads(args.resumir.read_text(encoding="utf-8"))), ensure_ascii=False, indent=2))
        return
    if not args.saida:
        parser.error("--gerar exige --saida")
    if args.saida.exists():
        parser.error("o arquivo de saída já existe; escolha outro para preservar a avaliação anterior")
    from app.embeddings import get_embedder
    from app.llm import get_llm
    base = json.loads(args.base.read_text(encoding="utf-8"))
    resultado = executar(base, get_embedder(), get_llm(config.LLM_MODEL))
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumir(resultado), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
