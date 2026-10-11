"""LLM externa, usada SOMENTE no nó de geração dos grafos.

O provedor é escolhido por variável de ambiente (LLM_PROVIDER / LLM_MODEL), sem
mudar código: groq (padrão, gratuito), openai, anthropic, google_genai...
Para provedores além do Groq, instale o pacote correspondente
(ex.: pip install langchain-anthropic) e defina a chave dele no .env.
"""

import os
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


class LLMLimiteAtingido(LLMIndisponivel):
    """O provedor recusou por limite de uso (HTTP 429). No plano gratuito da Groq
    o limite é por minuto, então basta esperar alguns segundos."""


class LLM(Protocol):
    def gerar(self, mensagens: list[BaseMessage]) -> str: ...


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
            )
        return self._modelo

    def gerar(self, mensagens: list[BaseMessage]) -> str:
        modelo = self._obter_modelo()
        try:
            resposta = modelo.invoke(mensagens)
        except Exception as erro:  # erros de rede, cota, chave inválida...
            if getattr(erro, "status_code", None) == 429:
                raise LLMLimiteAtingido(f"Limite de uso da LLM atingido ({config.LLM_PROVIDER}): {erro}") from erro
            raise LLMIndisponivel(f"Falha ao chamar a LLM ({config.LLM_PROVIDER}): {erro}") from erro
        conteudo = resposta.content
        if isinstance(conteudo, list):  # alguns provedores devolvem blocos
            conteudo = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in conteudo)
        texto = conteudo or ""
        if not texto.strip():
            raise LLMIndisponivel("A LLM devolveu uma resposta vazia.")
        return texto.strip()


@lru_cache(maxsize=4)
def get_llm(nome_modelo: str = config.LLM_MODEL) -> LLM:
    return LangChainLLM(nome_modelo)
