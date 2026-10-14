"""Resumo das avaliações (👍/👎) dadas às respostas na interface de demonstração.

As respostas com 👎 são as que mais ensinam: costumam ser perguntas que a base
não cobre ou em que a busca trouxe o trecho errado. Boas candidatas para entrar
em data/avaliacao/perguntas.json.

Uso (na raiz do projeto):  python -m scripts.resumo_feedback
"""

import json
from collections import Counter
from pathlib import Path

from app import config


def ler(arquivo: Path) -> list[dict]:
    if not arquivo.exists():
        return []
    with arquivo.open(encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]


def resumir(registros: list[dict]) -> dict:
    por_nota = Counter(r["rating"] for r in registros)
    por_perfil = {}
    for r in registros:
        contagem = por_perfil.setdefault(r.get("role", "STUDENT"), Counter())
        contagem[r["rating"]] += 1
    total = len(registros)
    return {
        "total": total,
        "positivas": por_nota["up"],
        "negativas": por_nota["down"],
        "aprovacao": por_nota["up"] / total if total else None,
        "por_perfil": {papel: dict(c) for papel, c in sorted(por_perfil.items())},
        "negativas_detalhe": [r for r in registros if r["rating"] == "down"],
    }


def main() -> None:
    resumo = resumir(ler(config.FEEDBACK_FILE))
    if not resumo["total"]:
        print(f"Nenhuma avaliação em {config.FEEDBACK_FILE}.")
        return
    print(f"{resumo['total']} avaliações: {resumo['positivas']} 👍, {resumo['negativas']} 👎 "
          f"({resumo['aprovacao'] * 100:.0f}% de aprovação)")
    for papel, contagem in resumo["por_perfil"].items():
        print(f"  {papel:<12} 👍 {contagem.get('up', 0):>3}   👎 {contagem.get('down', 0):>3}")
    if resumo["negativas_detalhe"]:
        print("\nRespostas com 👎:")
        for r in resumo["negativas_detalhe"]:
            fontes = ", ".join(r.get("sources") or []) or "sem fontes"
            print(f"  - {r['question']}  [{r.get('role')}; {fontes}]")
            if r.get("comment"):
                print(f"      comentário: {r['comment']}")


if __name__ == "__main__":
    main()
