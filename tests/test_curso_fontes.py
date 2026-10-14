from app.vector_store import Resultado


def test_aula_gerada_nao_usa_documento_fora_da_estrutura(embedder):
    from types import SimpleNamespace
    from app.graphs.curso_graph import _criar_escritor
    from tests.dubles import LLMFalsa

    outros = [Resultado("Conteúdo de outro documento", {"document": "outro.pdf", "page": 1}, 1.0)] * 25
    esperado = Resultado("Material selecionado", {"document": "manual.pdf", "page": 2}, 0.5)

    class Indice:
        index = SimpleNamespace(ntotal=26)

        def buscar(self, vetor, k):
            return (outros + [esperado])[:k]

    llm = LLMFalsa("Aula baseada no manual")
    escrever = _criar_escritor(llm, embedder, lambda *args: None, {"indice": Indice()})
    resultado = escrever({"structure": {"modules": [{
        "title": "Módulo", "objective": "Aprender", "lessons": [
            {"title": "Aula", "topics": ["Material"], "documents": ["manual.pdf"]},
        ],
    }]}})
    assert resultado["proposal"]["modules"][0]["lessons"][0]["sources"] == [{"document": "manual.pdf", "pages": "2"}]
    assert "Conteúdo de outro documento" not in llm.chamadas[0][1].content
