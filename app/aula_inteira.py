"""Perguntas de síntese sobre a aula aberta ("resuma esta aula", "quais os pontos
principais", "do que trata").

A busca por similaridade é a ferramenta errada para elas: "quais os pontos mais
importantes dessa aula" não se parece com nenhum trecho, e a busca acaba trazendo
um pedaço qualquer (um glossário, uma menção de passagem). Para sintetizar é
preciso ler a aula inteira. Tudo aqui é feito sem LLM: a regra da disciplina é que
a LLM externa só entra na geração da resposta final.
"""

import re

from app import config

_SINTESE = re.compile(
    r"\b("
    r"resum\w*|sintet\w*|s[ií]ntese|"
    r"pontos?\s+(mais\s+)?(importantes?|principais|centrais|chave)|principais\s+(pontos|ideias|t[oó]picos|assuntos|conceitos)|"
    r"ideias?\s+(principais|centrais)|"
    r"do\s+que\s+(se\s+)?trata|sobre\s+o\s+que\s+(é|e|fala|trata)|o\s+que\s+(essa|esta|a)\s+aula\s+(ensina|aborda|fala|diz)|"
    r"vis[aã]o\s+geral|panorama|"
    r"o\s+que\s+(eu\s+)?(preciso|devo)\s+(saber|aprender)\s+(dessa|desta|nessa|nesta|da)\s+aula"
    r")",
    re.IGNORECASE,
)


def e_pergunta_de_sintese(pergunta: str) -> bool:
    return bool(_SINTESE.search(pergunta or ""))


def _sem_titulo(texto: str, titulo: str | None) -> str:
    if titulo and texto.startswith(titulo + "\n"):
        return texto[len(titulo) + 1:]
    return texto


def reconstruir_texto(textos: list[str], titulo: str | None) -> str:
    """Junta os trechos da aula (na ordem) de volta num texto só, descontando a
    sobreposição entre trechos vizinhos e o título repetido no começo de cada um."""
    partes = [_sem_titulo(t, titulo).strip() for t in textos if t and t.strip()]
    if not partes:
        return ""
    texto = partes[0]
    for parte in partes[1:]:
        maximo = min(len(texto), len(parte), config.CHUNK_OVERLAP * 3)
        sobreposicao = next((n for n in range(maximo, 10, -1) if texto.endswith(parte[:n])), 0)
        texto += parte[sobreposicao:] if sobreposicao else "\n\n" + parte
    return texto


_ITEM_TITULO = re.compile(r"^(\d+(\.\d+)*\.?\s+\S.{0,110}|ANEXO\b.{0,110}|CAP[IÍ]TULO\b.{0,110})$")


def _primeira_frase(paragrafo: str, limite: int = 220) -> str:
    frase = re.split(r"(?<=[.;:])\s", paragrafo, maxsplit=1)[0]
    return frase if len(frase) <= limite else frase[:limite].rsplit(" ", 1)[0] + "…"


def condensar(texto: str, limite: int | None = None) -> tuple[str, bool]:
    """Devolve (texto, foi_condensado). Se a aula inteira couber no limite, vai
    inteira. Se não, vai uma versão condensada que cobre a aula toda, na ordem:
    os títulos das seções e a primeira frase de cada item. Nunca escolhe trechos
    por similaridade com a pergunta."""
    limite = limite or config.AULA_INTEIRA_MAX_CARACTERES
    if len(texto) <= limite:
        return texto, False

    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    titulos = {i for i, p in enumerate(paragrafos) if _ITEM_TITULO.match(p) and not p.endswith((".", ";", ":"))}
    resumo = {i: (p if i in titulos else _primeira_frase(p)) for i, p in enumerate(paragrafos)}

    # Os títulos das seções sempre entram; os demais itens entram espalhados
    # pela aula inteira até o limite, para cobrir do começo ao fim.
    escolhidos = set(titulos)
    tamanho = sum(len(resumo[i]) + 2 for i in titulos)
    demais = [i for i in range(len(paragrafos)) if i not in titulos]
    if demais and tamanho < limite:
        cabem = max(1, int((limite - tamanho) / (sum(len(resumo[i]) + 2 for i in demais) / len(demais))))
        passo = max(1.0, len(demais) / cabem)
        for n in range(min(cabem, len(demais))):
            i = demais[int(n * passo)]
            if tamanho + len(resumo[i]) + 2 > limite:
                break
            escolhidos.add(i)
            tamanho += len(resumo[i]) + 2

    condensado = "\n\n".join(resumo[i] for i in sorted(escolhidos))
    return condensado[:limite], True
