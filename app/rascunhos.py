"""Gerações de curso em andamento.

Gerar um curso leva minutos (várias chamadas à LLM), então o pedido volta na
hora e o trabalho segue em segundo plano. O Trainify consulta o andamento de
tempos em tempos. Uma geração por vez: várias ao mesmo tempo só disputariam o
limite por minuto da Groq.

Os rascunhos ficam em memória. Se o serviço reiniciar no meio, o Trainify recebe
404 e o instrutor pode pedir a geração de novo.
"""

import logging
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app import config

log = logging.getLogger("trainify-ai.rascunhos")

Executar = Callable[[dict, Callable[[int, str], None]], dict]


class FilaCheia(RuntimeError):
    pass


class GeracaoDescartada(RuntimeError):
    pass


class Rascunhos:
    def __init__(self, executar: Executar, pool: ThreadPoolExecutor | None = None):
        self._executar = executar
        self._pool = pool or ThreadPoolExecutor(max_workers=1, thread_name_prefix="geracao-curso")
        self._estados: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._pendentes = 0

    def criar(self, draft_id: str, pedido: dict) -> dict:
        with self._lock:
            atual = self._estados.get(draft_id)
            if atual:
                return deepcopy(atual)
            if self._pendentes >= config.CURSO_MAX_PENDENTES:
                raise FilaCheia("A fila de geração está cheia. Tente novamente em instantes.")
            estado = {
                "draft_id": draft_id, "status": "QUEUED", "progress": 0,
                "step": "Na fila", "result": None, "error": None, "created_at": time.time(),
            }
            self._estados[draft_id] = estado
            self._pendentes += 1
            self._descartar_antigos()
            resposta = deepcopy(estado)
        try:
            self._pool.submit(self._rodar, draft_id, estado, deepcopy(pedido))
        except Exception:
            with self._lock:
                self._pendentes -= 1
                if self._estados.get(draft_id) is estado:
                    self._estados.pop(draft_id)
            raise
        return self.obter(draft_id) or resposta

    def obter(self, draft_id: str) -> dict | None:
        with self._lock:
            estado = self._estados.get(draft_id)
            return deepcopy(estado) if estado else None

    def remover(self, draft_id: str) -> bool:
        with self._lock:
            return self._estados.pop(draft_id, None) is not None

    def _atualizar(self, draft_id: str, estado: dict, **campos) -> None:
        with self._lock:
            if self._estados.get(draft_id) is not estado:
                raise GeracaoDescartada()
            estado.update(campos)

    def _rodar(self, draft_id: str, estado: dict, pedido: dict) -> None:
        def progresso(percentual: int, etapa: str) -> None:
            campos = {"step": etapa}
            if percentual >= 0:  # -1 = só troca a mensagem (ex.: esperando o limite da IA)
                campos["progress"] = percentual
            self._atualizar(draft_id, estado, **campos)

        try:
            self._atualizar(draft_id, estado, status="PROCESSING", step="Começando")
            resultado = self._executar(pedido, progresso)
            self._atualizar(draft_id, estado, status="READY", progress=100, step="Pronto", result=resultado)
            log.info("Rascunho de curso %s pronto", draft_id)
        except GeracaoDescartada:
            pass
        except Exception as erro:  # o erro vai para o instrutor, o serviço continua de pé
            log.warning("Falha ao gerar o rascunho %s: %s", draft_id, erro)
            try:
                self._atualizar(draft_id, estado, status="FAILED", step="Falhou", error=_mensagem(erro))
            except GeracaoDescartada:
                pass
        finally:
            with self._lock:
                self._pendentes -= 1
                self._descartar_antigos()

    def _descartar_antigos(self) -> None:
        """Mantém só os rascunhos mais recentes (o resultado já foi salvo pelo Trainify)."""
        terminados = sorted(
            (e for e in self._estados.values() if e["status"] in ("READY", "FAILED")),
            key=lambda e: e["created_at"],
        )
        excesso = len(self._estados) - config.CURSO_RASCUNHOS_GUARDADOS
        for estado in terminados[:max(0, excesso)]:
            self._estados.pop(estado["draft_id"], None)


def _mensagem(erro: Exception) -> str:
    from app.graphs.curso_graph import EstruturaInvalida, MaterialInsuficiente
    from app.llm import LLMIndisponivel, LLMLimiteAtingido

    if isinstance(erro, MaterialInsuficiente):
        return str(erro)
    if isinstance(erro, LLMLimiteAtingido):
        return "O limite de uso da IA foi atingido várias vezes seguidas. Tente de novo em alguns minutos."
    if isinstance(erro, LLMIndisponivel):
        return "O serviço de IA está indisponível no momento. Tente de novo em instantes."
    if isinstance(erro, EstruturaInvalida):
        return "A IA não conseguiu montar uma estrutura de curso válida. Tente de novo ou envie menos documentos."
    return "Não foi possível gerar o curso. Tente de novo."
