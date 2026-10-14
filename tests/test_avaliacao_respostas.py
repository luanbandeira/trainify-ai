import json

from scripts.avaliar_respostas import BASE, CRITERIOS, executar, resumir
from tests.dubles import LLMFalsa


def test_base_tem_identificadores_unicos_e_escopos_validos():
    base = json.loads(BASE.read_text(encoding="utf-8"))
    assert len({c["id"] for c in base["cases"]}) == len(base["cases"])
    cursos = {m["course_id"] for m in base["materials"]}
    assert all(set(c["course_ids"]) <= cursos and c["expected_facts"] for c in base["cases"])


def test_execucao_nao_declara_qualidade_sem_revisao_humana(embedder):
    base = json.loads(BASE.read_text(encoding="utf-8"))
    base["cases"] = base["cases"][:1]
    resultado = executar(base, embedder, LLMFalsa("Resposta que precisa de revisão"))
    assert resultado["cases"][0]["error"] is None
    resumo = resumir(resultado)
    assert resumo["revisados"] == 0 and resumo["aprovacao_integral"] is None
    caso = resultado["cases"][0]
    caso["review"] = {c: True for c in CRITERIOS}
    assert resumir(resultado)["revisados"] == 0
    caso["reviewer"] = "Revisor de teste"
    caso["review"]["fundamentada"] = False
    assert resumir(resultado)["revisados"] == 1
    assert resumir(resultado)["aprovacao_integral"] == 0


def test_revisao_incompleta_nao_infla_resultado():
    resultado = {"cases": [{"error": None, "reviewer": "Revisor", "review": {"correta": True}}]}
    assert resumir(resultado)["revisados"] == 0
