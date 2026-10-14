import json

from app import config
from scripts.avaliar_rag import avaliar, recuperador_do_chat

PERGUNTAS = {
    "course_id": "curso-a",
    "no_assunto": [
        {"tipo": "conceito", "pergunta": "A partir de qual diferença de nível se aplica o trabalho em altura?", "aulas": ["nr-35"]},
        {"tipo": "conceito", "pergunta": "Quem deve fornecer o EPI gratuitamente ao trabalhador?", "aulas": ["nr-06"]},
        {"tipo": "coloquial", "pergunta": "quem guarda e conserva o equipamento de proteção individual?", "aulas": ["nr-35"]},
    ],
    "fora_do_assunto": ["Receita de bolo de chocolate"],
}


def test_metricas_da_busca(store, embedder):
    resultado = avaliar(PERGUNTAS, recuperador_do_chat(store, embedder))

    geral = resultado["geral"]
    assert geral["perguntas"] == 3
    # as duas primeiras acham a aula certa em 1º; a terceira espera de propósito a aula errada
    assert geral["acerto@1"] == 2 / 3
    assert resultado["casos"][2]["primeira"] == "nr-06"
    assert set(resultado["por_tipo"]) == {"coloquial", "conceito"}
    assert resultado["por_tipo"]["conceito"]["acerto@1"] == 1
    assert resultado["recusa_correta"] == 1


def test_busca_so_no_curso_avaliado(store, embedder):
    # o curso-b fala de "trabalho em altura e EPI", mas não pode aparecer na avaliação do curso-a
    resultado = avaliar(PERGUNTAS, recuperador_do_chat(store, embedder))
    assert all(c["primeira"] in {"nr-35", "nr-06", None} for c in resultado["casos"])


def test_arquivo_de_perguntas_e_valido():
    perguntas = json.loads((config.DATA_DIR / "avaliacao" / "perguntas.json").read_text(encoding="utf-8"))
    curso = json.loads((config.DATA_DIR / "curso_nrs.json").read_text(encoding="utf-8"))
    aulas = {licao["lesson_id"] for modulo in curso["modules"] for licao in modulo["lessons"]}

    assert perguntas["course_id"] == curso["course_id"]
    assert perguntas["no_assunto"] and perguntas["fora_do_assunto"]
    for caso in perguntas["no_assunto"]:
        assert set(caso["aulas"]) <= aulas, caso
