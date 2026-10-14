import json

import pytest
from fastapi.testclient import TestClient

from app import config, main
from scripts.resumo_feedback import ler, resumir


@pytest.fixture
def arquivo(tmp_path, monkeypatch):
    caminho = tmp_path / "feedback" / "feedback.jsonl"
    monkeypatch.setattr(config, "FEEDBACK_FILE", caminho)
    return caminho


@pytest.fixture
def cliente():
    return TestClient(main.app)


def _avaliar(cliente, rating, **extra):
    return cliente.post("/demo/feedback", json={
        "rating": rating, "question": "Quem fornece o EPI?", "answer": "A organização.", **extra,
    })


def test_feedback_e_gravado_uma_linha_por_avaliacao(cliente, arquivo):
    assert _avaliar(cliente, "up", sources=["nr-06"]).status_code == 201
    assert _avaliar(cliente, "down", role="MANAGER", comment="faltou o item").status_code == 201

    linhas = [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines()]
    assert [l["rating"] for l in linhas] == ["up", "down"]
    assert linhas[0]["sources"] == ["nr-06"]
    assert linhas[1]["comment"] == "faltou o item"
    assert "at" in linhas[0]


def test_feedback_invalido_e_recusado(cliente, arquivo):
    assert _avaliar(cliente, "talvez").status_code == 422
    assert _avaliar(cliente, "up", comment="x" * 501).status_code == 422
    assert not arquivo.exists()


def test_feedback_so_no_modo_demo(cliente, arquivo, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", False)
    assert _avaliar(cliente, "up").status_code == 404


def test_resumo_do_feedback(cliente, arquivo):
    _avaliar(cliente, "up")
    _avaliar(cliente, "up", role="MANAGER")
    _avaliar(cliente, "down", sources=["nr-35"])

    resumo = resumir(ler(arquivo))
    assert (resumo["total"], resumo["positivas"], resumo["negativas"]) == (3, 2, 1)
    assert resumo["aprovacao"] == 2 / 3
    assert resumo["por_perfil"] == {"MANAGER": {"up": 1}, "STUDENT": {"up": 1, "down": 1}}
    assert resumo["negativas_detalhe"][0]["sources"] == ["nr-35"]


def test_resumo_sem_arquivo(tmp_path):
    assert resumir(ler(tmp_path / "nada.jsonl"))["total"] == 0
