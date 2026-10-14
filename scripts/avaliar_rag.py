"""Avaliação da busca do RAG (a etapa de recuperação, sem LLM).

Roda as perguntas de data/avaliacao/perguntas.json pelo mesmo nó
recuperar_contexto do grafo de chat e mede:

  - acerto@1:  o primeiro trecho é de uma aula que responde a pergunta
  - acerto@k:  alguma aula certa aparece entre os TOP_K trechos que vão para o prompt
  - MRR:       média de 1/posição do primeiro trecho certo (0 quando não aparece)
  - recusa indevida:  pergunta do assunto que ficou sem nenhum trecho (o chat diria
                      que o material não cobre o assunto)
  - recusa correta:   pergunta fora do assunto que ficou sem trecho (a LLM nem é chamada)

Não usa a LLM nem precisa de chave: só o índice e o modelo de embeddings.

Uso (na raiz do projeto, depois de python -m scripts.indexar_nrs):
    python -m scripts.avaliar_rag
    python -m scripts.avaliar_rag --detalhes     # mostra cada pergunta
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app import config
from app.graphs.chat_graph import _criar_recuperador

ARQUIVO_PADRAO = config.DATA_DIR / "avaliacao" / "perguntas.json"


def _posicao_do_acerto(contextos, aulas_certas: set[str]) -> int | None:
    for posicao, r in enumerate(contextos, start=1):
        if r.metadata.get("lesson_id") in aulas_certas:
            return posicao
    return None


def avaliar(perguntas: dict, recuperar) -> dict:
    """`recuperar(pergunta, course_id)` devolve a lista de trechos que iria para o prompt."""
    course_id = perguntas["course_id"]
    casos = []
    for caso in perguntas["no_assunto"]:
        contextos = recuperar(caso["pergunta"], course_id)
        posicao = _posicao_do_acerto(contextos, set(caso["aulas"]))
        casos.append({
            **caso,
            "posicao": posicao,
            "recusada": not contextos,
            "primeira": contextos[0].metadata.get("lesson_id") if contextos else None,
            "nota": round(contextos[0].score, 3) if contextos else None,
        })

    fora = []
    for pergunta in perguntas["fora_do_assunto"]:
        contextos = recuperar(pergunta, course_id)
        fora.append({
            "pergunta": pergunta,
            "recusada": not contextos,
            "nota": round(contextos[0].score, 3) if contextos else None,
        })

    def metricas(grupo: list[dict]) -> dict:
        n = len(grupo) or 1
        return {
            "perguntas": len(grupo),
            "acerto@1": sum(c["posicao"] == 1 for c in grupo) / n,
            "acerto@k": sum(c["posicao"] is not None for c in grupo) / n,
            "mrr": sum(1 / c["posicao"] for c in grupo if c["posicao"]) / n,
            "recusa_indevida": sum(c["recusada"] for c in grupo) / n,
        }

    por_tipo = defaultdict(list)
    for c in casos:
        por_tipo[c["tipo"]].append(c)

    return {
        "geral": metricas(casos),
        "por_tipo": {tipo: metricas(grupo) for tipo, grupo in sorted(por_tipo.items())},
        "recusa_correta": sum(f["recusada"] for f in fora) / (len(fora) or 1),
        "casos": casos,
        "fora_do_assunto": fora,
    }


def recuperador_do_chat(store, embedder):
    """O mesmo nó de recuperação que o grafo de chat usa."""
    no = _criar_recuperador(store, embedder)

    def recuperar(pergunta: str, course_id: str):
        return no({"search_query": pergunta, "course_ids": [course_id]})["contexts"]

    return recuperar


def _pct(valor: float) -> str:
    return f"{valor * 100:5.1f}%"


def imprimir(resultado: dict, detalhes: bool = False) -> None:
    print(f"{'grupo':<12} {'n':>3} {'acerto@1':>9} {'acerto@k':>9} {'MRR':>6} {'recusa indevida':>16}")
    linhas = [("geral", resultado["geral"]), *resultado["por_tipo"].items()]
    for nome, m in linhas:
        print(f"{nome:<12} {m['perguntas']:>3} {_pct(m['acerto@1']):>9} {_pct(m['acerto@k']):>9} "
              f"{m['mrr']:>6.3f} {_pct(m['recusa_indevida']):>16}")
    print(f"\nPerguntas fora do assunto recusadas: {_pct(resultado['recusa_correta']).strip()}"
          f" ({sum(f['recusada'] for f in resultado['fora_do_assunto'])}/{len(resultado['fora_do_assunto'])})")

    if detalhes:
        print("\nNo assunto:")
        for c in resultado["casos"]:
            marca = "ok " if c["posicao"] == 1 else ("~  " if c["posicao"] else "X  ")
            onde = "recusada" if c["recusada"] else f"1º: {c['primeira']} ({c['nota']})"
            print(f"  {marca} [{c['tipo']}] {c['pergunta']}  ->  esperado {','.join(c['aulas'])}; {onde}")
        print("\nFora do assunto:")
        for f in resultado["fora_do_assunto"]:
            print(f"  {'ok ' if f['recusada'] else 'X  '} {f['pergunta']}  (melhor nota: {f['nota']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--perguntas", type=Path, default=ARQUIVO_PADRAO)
    parser.add_argument("--detalhes", action="store_true", help="mostra o resultado de cada pergunta")
    parser.add_argument("--json", type=Path, help="grava o resultado completo neste arquivo")
    args = parser.parse_args()

    from app.embeddings import get_embedder
    from app.vector_store import VectorStore

    perguntas = json.loads(args.perguntas.read_text(encoding="utf-8"))
    embedder = get_embedder()
    store = VectorStore(embedder.dimension)
    if not store.existe(perguntas["course_id"]):
        raise SystemExit(f"Curso {perguntas['course_id']} sem índice. Rode antes: python -m scripts.indexar_nrs")

    print(f"TOP_K={config.TOP_K}  MIN_SCORE={config.MIN_SCORE}\n")
    resultado = avaliar(perguntas, recuperador_do_chat(store, embedder))
    imprimir(resultado, args.detalhes)
    if args.json:
        args.json.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
