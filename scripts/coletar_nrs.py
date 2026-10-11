"""Etapa 1 do pipeline: coleta.

Baixa do gov.br o PDF mais recente de cada NR listada em data/curso_nrs.json e
salva em data/raw/, junto de um manifest.json com a origem de cada arquivo.

Uso (na raiz do projeto):  python -m scripts.coletar_nrs
"""

import json
import re
import sys
import time
from datetime import date
from urllib.parse import urljoin

import httpx

from app.config import DATA_DIR, RAW_DIR

LISTAGEM = (
    "https://www.gov.br/trabalho-e-emprego/pt-br/acesso-a-informacao/participacao-social/"
    "conselhos-e-orgaos-colegiados/comissao-tripartite-partitaria-permanente/"
    "normas-regulamentadora/normas-regulamentadoras-vigentes"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (trainify-ai; projeto academico)"}


def carregar_curso() -> dict:
    return json.loads((DATA_DIR / "curso_nrs.json").read_text(encoding="utf-8"))


def pagina_da_nr(html_listagem: str, numero: int) -> str | None:
    """Endereço da página de uma NR na listagem de normas vigentes."""
    padroes = [rf'href="([^"]*norma-regulamentadora-no-{numero}-nr-{numero})"', rf'href="([^"]*/nr-{numero})"']
    for padrao in padroes:
        achado = re.search(padrao, html_listagem)
        if achado:
            return achado.group(1)
    return None


def pdf_mais_recente(html_pagina: str, base: str, numero: int) -> str | None:
    """Escolhe o texto consolidado mais recente: arquivos nr-XX-atualizada-ANO*.pdf."""
    candidatos = []
    for href in re.findall(r'href="([^"]+\.pdf)"', html_pagina):
        nome = href.rsplit("/", 1)[-1].lower()
        achado = re.match(rf"nr-?0?{numero}-atualizad[ao]-(\d{{4}})", nome)
        if achado:
            candidatos.append((int(achado.group(1)), nome, urljoin(base, href)))
    if not candidatos:
        return None
    # Maior ano; empate desfeito pelo nome (sufixos como -i-3 vêm depois de -i-1)
    candidatos.sort()
    return candidatos[-1][2]


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    curso = carregar_curso()
    licoes = [l for m in curso["modules"] for l in m["lessons"]]

    manifest = []
    falhas = []
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as http:
        listagem = http.get(LISTAGEM).text

        for licao in licoes:
            numero = licao["nr"]
            destino = RAW_DIR / f"{licao['lesson_id']}.pdf"

            pagina = pagina_da_nr(listagem, numero)
            if not pagina:
                falhas.append(f"NR-{numero}: página não encontrada na listagem")
                continue

            url_pdf = pdf_mais_recente(http.get(pagina).text, pagina, numero)
            if not url_pdf:
                falhas.append(f"NR-{numero}: PDF consolidado não encontrado em {pagina}")
                continue

            if not destino.exists():
                resposta = http.get(url_pdf)
                resposta.raise_for_status()
                if not resposta.content.startswith(b"%PDF"):
                    falhas.append(f"NR-{numero}: o endereço não devolveu um PDF ({url_pdf})")
                    continue
                destino.write_bytes(resposta.content)
                time.sleep(1)  # educação com o servidor do governo

            print(f"NR-{numero:>2}: {destino.name} ({destino.stat().st_size // 1024} KB) <- {url_pdf}")
            manifest.append({
                "lesson_id": licao["lesson_id"],
                "nr": numero,
                "title": licao["title"],
                "source_url": url_pdf,
                "page_url": pagina,
                "file": destino.name,
                "collected_at": date.today().isoformat(),
            })

    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} de {len(licoes)} normas coletadas em {RAW_DIR}")
    for falha in falhas:
        print("  FALHA:", falha)
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
