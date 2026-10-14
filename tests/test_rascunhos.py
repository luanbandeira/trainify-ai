import pytest

from app import config
from app.rascunhos import FilaCheia, Rascunhos


class PoolControlado:
    def __init__(self):
        self.trabalhos = []

    def submit(self, funcao, *args):
        self.trabalhos.append(lambda: funcao(*args))

    def rodar(self):
        self.trabalhos.pop(0)()


def test_requisicao_repetida_nao_gera_novamente_apos_conclusao():
    pool = PoolControlado()
    r = Rascunhos(lambda pedido, progresso: {"title": "Curso"}, pool)
    r.criar("a", {})
    pool.rodar()
    assert r.criar("a", {})["status"] == "READY"
    assert not pool.trabalhos
    copia = r.obter("a")
    copia["result"]["title"] = "Alterado"
    assert r.obter("a")["result"]["title"] == "Curso"


def test_descarte_nao_libera_capacidade_antes_de_encerrar_trabalho(monkeypatch):
    monkeypatch.setattr(config, "CURSO_MAX_PENDENTES", 1)
    pool = PoolControlado()
    r = Rascunhos(lambda pedido, progresso: {}, pool)
    r.criar("a", {})
    r.remover("a")
    with pytest.raises(FilaCheia):
        r.criar("b", {})
    pool.rodar()
    assert r.criar("b", {})["status"] == "QUEUED"


def test_geracao_antiga_nao_sobrescreve_rascunho_recriado():
    pool = PoolControlado()

    def executar(pedido, progresso):
        if pedido["versao"] == 1:
            r.remover("a")
            r.criar("a", {"versao": 2})
        return pedido

    r = Rascunhos(executar, pool)
    r.criar("a", {"versao": 1})
    pool.rodar()
    assert r.obter("a")["status"] == "QUEUED"
    assert r.obter("a")["result"] is None
    pool.rodar()
    assert r.obter("a")["result"] == {"versao": 2}


def test_falha_ao_agendar_libera_vaga(monkeypatch):
    monkeypatch.setattr(config, "CURSO_MAX_PENDENTES", 1)

    class PoolFalhaUmaVez(PoolControlado):
        falhar = True

        def submit(self, funcao, *args):
            if self.falhar:
                self.falhar = False
                raise RuntimeError("Executor indisponível")
            super().submit(funcao, *args)

    r = Rascunhos(lambda pedido, progresso: {}, PoolFalhaUmaVez())
    with pytest.raises(RuntimeError):
        r.criar("a", {})
    assert r.obter("a") is None
    assert r.criar("b", {})["status"] == "QUEUED"
