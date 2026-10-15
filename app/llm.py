"""LLM externa, usada SOMENTE no nó de geração dos grafos.

O provedor é escolhido por variável de ambiente (LLM_PROVIDER / LLM_MODEL), sem
mudar código: groq (padrão, gratuito), openai, anthropic, google_genai...
Para provedores além do Groq, instale o pacote correspondente
(ex.: pip install langchain-anthropic) e defina a chave dele no .env.
"""

import os
from collections.abc import Iterator
from functools import lru_cache
from typing import Protocol

from langchain_core.messages import BaseMessage

from app import config

_CHAVES = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}


class LLMIndisponivel(RuntimeError):
    """A LLM não está configurada ou falhou ao responder."""


class LLMRespostaVazia(LLMIndisponivel):
    """A LLM terminou sem texto. Acontece de vez em quando com modelos que
    "pensam" antes de responder; tentar de novo costuma resolver."""


class LLMRespostaCortada(LLMRespostaVazia):
    """A resposta parou no limite de tamanho, no meio de uma frase. Guarda o texto
    parcial: no chat ele ainda serve; na geração de curso, tenta-se de novo."""

    def __init__(self, mensagem: str, parcial: str):
        super().__init__(mensagem)
        self.parcial = parcial


class LLMLimiteAtingido(LLMIndisponivel):
    """O provedor recusou por limite de uso (HTTP 429). No plano gratuito da Groq
    o limite é por minuto, então basta esperar alguns segundos."""


class LLM(Protocol):
    def gerar(self, mensagens: list[BaseMessage]) -> str: ...


class LLMComStreaming(LLM, Protocol):
    def gerar_stream(self, mensagens: list[BaseMessage]) -> Iterator[str]: ...


def llm_configurada() -> bool:
    chave = _CHAVES.get(config.LLM_PROVIDER)
    return bool(os.getenv(chave)) if chave else True


class LangChainLLM:
    """O modelo só é criado na primeira geração: quando a busca não acha contexto,
    o grafo responde sem chamar a LLM, e isso funciona mesmo sem chave configurada."""

    def __init__(self, nome_modelo: str = config.LLM_MODEL):
        self.nome_modelo = nome_modelo
        self._modelo = None

    def _obter_modelo(self):
        if self._modelo is None:
            if not llm_configurada():
                raise LLMIndisponivel(
                    f"Defina {_CHAVES[config.LLM_PROVIDER]} no arquivo .env para usar o provedor {config.LLM_PROVIDER}."
                )
            from langchain.chat_models import init_chat_model

            self._modelo = init_chat_model(
                self.nome_modelo,
                model_provider=config.LLM_PROVIDER,
                temperature=config.LLM_TEMPERATURE,
                timeout=config.LLM_TIMEOUT_SECONDS,
                # Os SDKs respeitam o Retry-After do 429, então tentar de novo resolve
                # a maioria dos estouros momentâneos do limite por minuto.
                max_retries=config.LLM_MAX_RETRIES,
                # Os modelos gpt-oss "pensam" antes de responder e esse raciocínio
                # também conta no limite; sem folga, aulas longas saíam cortadas
                max_tokens=config.LLM_MAX_TOKENS,
            )
        return self._modelo

    @staticmethod
    def _erro_do_provedor(erro: Exception) -> LLMIndisponivel:
        if getattr(erro, "status_code", None) == 429:
            return LLMLimiteAtingido(f"Limite de uso da LLM atingido ({config.LLM_PROVIDER}): {erro}")
        return LLMIndisponivel(f"Falha ao chamar a LLM ({config.LLM_PROVIDER}): {erro}")

    @staticmethod
    def _texto(conteudo) -> str:
        if isinstance(conteudo, list):  # alguns provedores devolvem blocos
            conteudo = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in conteudo)
        return conteudo or ""

    def gerar(self, mensagens: list[BaseMessage]) -> str:
        modelo = self._obter_modelo()
        try:
            resposta = modelo.invoke(mensagens)
        except Exception as erro:  # erros de rede, cota, chave inválida...
            raise self._erro_do_provedor(erro) from erro
        texto = self._texto(resposta.content)
        if not texto.strip():
            raise LLMRespostaVazia("A LLM devolveu uma resposta vazia.")
        if (getattr(resposta, "response_metadata", None) or {}).get("finish_reason") == "length":
            raise LLMRespostaCortada("A resposta da LLM foi cortada no limite de tamanho.", texto.strip())
        return texto.strip()

    def gerar_stream(self, mensagens: list[BaseMessage]) -> Iterator[str]:
        """Como gerar(), mas entrega o texto aos pedaços, conforme a LLM escreve.
        Os mesmos erros valem: vazia, cortada (com o parcial) e limite atingido."""
        modelo = self._obter_modelo()
        texto, fim = "", None
        fluxo = None
        try:
            fluxo = modelo.stream(mensagens)
            for pedaco in fluxo:
                parte = self._texto(pedaco.content)
                fim = (getattr(pedaco, "response_metadata", None) or {}).get("finish_reason") or fim
                if parte:
                    texto += parte
                    yield parte
        except Exception as erro:
            raise self._erro_do_provedor(erro) from erro
        finally:
            fechar = getattr(fluxo, "close", None)
            if fechar:
                fechar()
        if not texto.strip():
            raise LLMRespostaVazia("A LLM devolveu uma resposta vazia.")
        if fim == "length":
            raise LLMRespostaCortada("A resposta da LLM foi cortada no limite de tamanho.", texto.strip())


@lru_cache(maxsize=4)
def get_llm(nome_modelo: str = config.LLM_MODEL) -> LLM:
    return LangChainLLM(nome_modelo)
