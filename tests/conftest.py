import pytest

from app import config
from app.ingestao import Aula, indexar_aula
from app.vector_store import VectorStore
from tests.dubles import EmbedderFalso

ALTURA = (
    "35.2.1 Esta Norma se aplica a toda atividade com diferença de nível acima de 2,0 m (dois metros) "
    "do nível inferior, onde haja risco de queda. 35.3.1 Cabe à organização garantir a implementação "
    "das medidas de prevenção e assegurar a realização da Análise de Risco antes do trabalho em altura."
)
EPI = (
    "6.5.1 Cabe à organização fornecer ao trabalhador, gratuitamente, EPI adequado ao risco, em perfeito "
    "estado de conservação e funcionamento. 6.5.2 Cabe ao trabalhador usar o EPI apenas para a finalidade "
    "a que se destina e responsabilizar-se pela guarda e conservação do equipamento de proteção individual."
)


@pytest.fixture(autouse=True)
def _config_de_teste(monkeypatch):
    # Os dublês não reproduzem a escala de notas do e5; o corte fica baixo.
    monkeypatch.setattr(config, "MIN_SCORE", 0.2)
    monkeypatch.setattr(config, "CHUNK_SIZE", 300)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 30)
    monkeypatch.setattr(config, "INTERNAL_API_KEY", "")
    monkeypatch.setattr(config, "PRELOAD_EMBEDDINGS", False)


@pytest.fixture
def embedder():
    return EmbedderFalso()


@pytest.fixture
def store(tmp_path, embedder):
    store = VectorStore(embedder.dimension, pasta_base=tmp_path / "indexes")
    indexar_aula(Aula(course_id="curso-a", lesson_id="nr-35", module_id="m1", title="NR-35 Trabalho em altura",
                      content=ALTURA, tenant_id="empresa-a"), store, embedder)
    indexar_aula(Aula(course_id="curso-a", lesson_id="nr-06", module_id="m1", title="NR-6 EPI",
                      content=EPI, tenant_id="empresa-a"), store, embedder)
    indexar_aula(Aula(course_id="curso-b", lesson_id="segredo", title="Curso de outra empresa",
                      content="Plano estratégico confidencial da empresa B para trabalho em altura e EPI. " * 3,
                      tenant_id="empresa-b"), store, embedder)
    return store
