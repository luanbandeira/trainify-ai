"""Etapas 2 e 3 do pipeline: limpeza e chunking.

Serve tanto para a base de NRs quanto para o material que o instrutor sobe no
Trainify (texto da aula, PDF, transcrição do vídeo).
"""

import io
import re
from collections import Counter
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app import config


def extrair_texto_pdf(conteudo: bytes) -> list[str]:
    """Texto de cada página do PDF, na ordem."""
    if not conteudo.startswith(b"%PDF"):
        raise ValueError("O arquivo enviado não é um PDF")
    try:
        leitor = PdfReader(io.BytesIO(conteudo))
        if leitor.is_encrypted:
            raise ValueError("Envie um PDF sem proteção por senha")
        if len(leitor.pages) > config.MAX_PDF_PAGES:
            raise ValueError(f"O PDF excede o limite de {config.MAX_PDF_PAGES} páginas")
        paginas = []
        tamanho = 0
        for pagina in leitor.pages:
            texto = pagina.extract_text() or ""
            tamanho += len(texto)
            if tamanho > config.MAX_TEXT_CHARS:
                raise ValueError("O texto do PDF excede o limite permitido")
            paginas.append(texto)
        return paginas
    except PyPdfError as erro:
        raise ValueError("Não foi possível ler o PDF. Envie um arquivo válido e sem proteção por senha") from erro


def _linhas_repetidas(paginas: list[str], minimo: int = 3) -> set[str]:
    """Cabeçalhos e rodapés: linhas curtas que aparecem em boa parte das páginas."""
    if len(paginas) < minimo:
        return set()
    contagem = Counter()
    for pagina in paginas:
        linhas = {l.strip() for l in pagina.splitlines() if 0 < len(l.strip()) <= 120}
        contagem.update(linhas)
    limite = max(minimo, len(paginas) // 3)
    return {linha for linha, vezes in contagem.items() if vezes >= limite}


_NUMERO_DE_PAGINA = re.compile(r"^(p[aá]gina\s*)?\d{1,4}(\s*(de|/)\s*\d{1,4})?$", re.IGNORECASE)
# Linha que começa um novo item da norma: "12.1.2", "a)", "I -", "ANEXO II", "Tabela 1" ...
_INICIO_DE_ITEM = re.compile(
    r"^(\d+(\.\d+)+\.?\s|[a-z]\)\s|[IVXLC]+\s*[-–]\s|ANEXO\b|CAP[IÍ]TULO\b|SUM[AÁ]RIO\b|Tabela\b|Quadro\b|Nota\b|NR-\d+)",
)


def limpar_paginas(paginas: list[str]) -> str:
    """Remove cabeçalhos, rodapés e números de página e refaz os parágrafos."""
    repetidas = _linhas_repetidas(paginas)
    linhas_uteis: list[str] = []

    for pagina in paginas:
        for linha in pagina.splitlines():
            linha = re.sub(r"\s+", " ", linha).strip()
            if not linha or linha in repetidas or _NUMERO_DE_PAGINA.match(linha):
                continue
            linhas_uteis.append(linha)

    return remover_ruido_normativo(juntar_linhas(linhas_uteis))


_HISTORICO_DOU = re.compile(r"\s*Publica[cç][aã]o D\.O\.U\..*$", re.DOTALL)


def remover_ruido_normativo(texto: str) -> str:
    """Tira o que não ajuda a responder perguntas em textos de normas: a lista de
    portarias que alteraram a norma e o sumário (que repete os títulos das seções
    e geraria trechos sem conteúdo)."""
    paragrafos = []
    for p in texto.split("\n\n"):
        p = p.replace("Este texto não substitui o publicado no DOU", "").strip()
        if "Alterações/Atualizações D.O.U." in p or p.startswith("Publicação D.O.U."):
            p = _HISTORICO_DOU.sub("", p).strip()  # mantém só o título da norma
        if p:
            paragrafos.append(p)

    def normalizar(s: str) -> str:
        return re.sub(r"[^\w]", "", s.lower())

    inicio = next((i for i, p in enumerate(paragrafos) if p.upper().startswith("SUMÁRIO")), None)
    if inicio is not None and inicio + 1 < len(paragrafos):
        # O sumário termina quando o primeiro título dele reaparece no corpo da norma.
        # Se não reaparecer, não arriscamos cortar nada.
        primeiro = normalizar(paragrafos[inicio + 1])
        fim = next(
            (j for j in range(inicio + 2, len(paragrafos)) if normalizar(paragrafos[j]) == primeiro), None
        )
        if fim is not None:
            paragrafos = paragrafos[:inicio] + paragrafos[fim:]
    return "\n\n".join(paragrafos)


def juntar_linhas(linhas: list[str]) -> str:
    """O PDF quebra o texto por largura da página; aqui as linhas voltam a formar
    parágrafos, e cada item numerado da norma vira um parágrafo próprio."""
    paragrafos: list[str] = []
    atual = ""
    for linha in linhas:
        if not atual:
            atual = linha
        elif _INICIO_DE_ITEM.match(linha):
            paragrafos.append(atual)
            atual = linha
        elif atual.endswith("-") and not atual.endswith(" -"):
            atual = atual[:-1] + linha  # palavra hifenizada na quebra de linha
        else:
            atual = f"{atual} {linha}"
    if atual:
        paragrafos.append(atual)
    return "\n\n".join(paragrafos)


def paginas_limpas(paginas: list[str]) -> list[str]:
    """Como limpar_paginas, mas devolve o texto de cada página separado, para os
    trechos guardarem de qual página vieram."""
    repetidas = _linhas_repetidas(paginas)
    resultado = []
    for pagina in paginas:
        linhas = []
        for linha in pagina.splitlines():
            linha = re.sub(r"\s+", " ", linha).strip()
            if linha and linha not in repetidas and not _NUMERO_DE_PAGINA.match(linha):
                linhas.append(linha)
        resultado.append(juntar_linhas(linhas))
    return resultado


def limpar_texto(texto: str) -> str:
    """Limpeza leve para texto que já vem digitado (conteúdo da aula, transcrição)."""
    texto = texto.replace("\r\n", "\n").replace(" ", " ")
    texto = re.sub(r"<[^>]+>", " ", texto)          # HTML do editor de texto
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


@dataclass
class Chunk:
    text: str
    metadata: dict


def dividir_em_chunks(texto: str, metadata: dict, titulo: str | None = None) -> list[Chunk]:
    """Quebra o texto em trechos com sobreposição. Cada trecho recebe o título da
    aula como prefixo, para não perder o assunto quando é lido fora de contexto."""
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    partes = [p.strip() for p in divisor.split_text(texto) if len(p.strip()) >= 40]
    prefixo = f"{titulo}\n" if titulo else ""
    return [
        Chunk(text=prefixo + parte, metadata={**metadata, "chunk_index": i})
        for i, parte in enumerate(partes)
    ]


def trechos_por_pagina(paginas: list[str], metadata: dict, titulo: str | None = None) -> list[Chunk]:
    """Divide página por página, para cada trecho saber o número da página de onde
    veio (usado para mostrar as fontes de uma aula gerada pela IA)."""
    trechos: list[Chunk] = []
    for numero, texto in enumerate(paginas, start=1):
        for chunk in dividir_em_chunks(texto, {**metadata, "page": numero}, titulo=titulo):
            chunk.metadata["chunk_index"] = len(trechos)
            trechos.append(chunk)
    return trechos
