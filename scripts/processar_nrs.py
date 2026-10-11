"""Etapa 2 do pipeline: limpeza.

Extrai o texto dos PDFs em data/raw/ e grava a versão limpa em data/processed/.

Uso (na raiz do projeto):  python -m scripts.processar_nrs
"""

import json

from app.config import PROCESSED_DIR, RAW_DIR
from app.text_processing import extrair_texto_pdf, limpar_paginas


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))

    total_paginas = total_palavras = 0
    for item in manifest:
        paginas = extrair_texto_pdf((RAW_DIR / item["file"]).read_bytes())
        texto = limpar_paginas(paginas)
        (PROCESSED_DIR / f"{item['lesson_id']}.txt").write_text(texto, encoding="utf-8")

        palavras = len(texto.split())
        total_paginas += len(paginas)
        total_palavras += palavras
        print(f"{item['lesson_id']}: {len(paginas):>3} páginas, {palavras:>6} palavras")

    print(f"\nTotal: {len(manifest)} normas, {total_paginas} páginas de PDF, {total_palavras} palavras")


if __name__ == "__main__":
    main()
