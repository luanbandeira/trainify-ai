"""Configuração do serviço, lida das variáveis de ambiente (arquivo .env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _int(nome: str, padrao: int) -> int:
    return int(os.getenv(nome, padrao))


def _float(nome: str, padrao: float) -> float:
    return float(os.getenv(nome, padrao))


# ── Pastas de dados ──
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"              # PDFs coletados
PROCESSED_DIR = DATA_DIR / "processed"  # textos limpos
INDEX_DIR = DATA_DIR / "indexes"        # um índice FAISS por curso

# ── LLM externa (usada apenas na geração) ──
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.2)
LLM_TIMEOUT_SECONDS = _int("LLM_TIMEOUT_SECONDS", 60)
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 4)
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 4096)  # inclui o raciocínio dos modelos gpt-oss
# O quiz é gerado poucas vezes e precisa de mais precisão (alternativas erradas
# que sejam de fato falsas), então usa um modelo maior que o do chat.
QUIZ_LLM_MODEL = os.getenv("QUIZ_LLM_MODEL", "openai/gpt-oss-120b")

# ── Embeddings locais ──
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
# Carregar o modelo ao subir o serviço (desligado nos testes, que usam dublês)
PRELOAD_EMBEDDINGS = os.getenv("PRELOAD_EMBEDDINGS", "true").lower() == "true"

# ── Chunking ──
CHUNK_SIZE = _int("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)

# ── Recuperação ──
TOP_K = _int("TOP_K", 6)
# Similaridade de cosseno mínima para um trecho entrar no prompt. O e5 dá notas
# altas mesmo para textos sem relação (0,77 a 0,84 nas perguntas fora do assunto
# que testamos na base de NRs), enquanto perguntas do domínio, mesmo coloquiais,
# ficaram acima de 0,845. Veja a seção "Calibração" do README.
MIN_SCORE = _float("MIN_SCORE", 0.845)
HISTORY_MESSAGES = _int("HISTORY_MESSAGES", 6)
# Busca híbrida: o BM25 (palavras-chave) soma até HYBRID_WEIGHT à similaridade na
# hora de ordenar os trechos. Calibrado com `python -m scripts.avaliar_rag`.
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "true").lower() == "true"
HYBRID_WEIGHT = _float("HYBRID_WEIGHT", 0.03)

# ── Quiz ──
QUIZ_MAX_CHUNKS = _int("QUIZ_MAX_CHUNKS", 12)
QUIZ_MAX_ATTEMPTS = _int("QUIZ_MAX_ATTEMPTS", 3)

# ── Geração de curso a partir de documentos ──
CURSO_MAX_DOCUMENTOS = _int("CURSO_MAX_DOCUMENTOS", 10)
CURSO_MAPA_CARACTERES = _int("CURSO_MAPA_CARACTERES", 14000)   # tamanho do "mapa" enviado para propor a estrutura
CURSO_MAPA_AMOSTRA = _int("CURSO_MAPA_AMOSTRA", 220)           # caracteres de cada amostra do mapa
CURSO_TRECHOS_POR_AULA = _int("CURSO_TRECHOS_POR_AULA", 6)
CURSO_TENTATIVAS_LIMITE = _int("CURSO_TENTATIVAS_LIMITE", 8)   # esperas pelo limite por minuto da Groq
CURSO_ESPERA_LIMITE_S = _int("CURSO_ESPERA_LIMITE_S", 20)
CURSO_RASCUNHOS_GUARDADOS = _int("CURSO_RASCUNHOS_GUARDADOS", 30)
CURSO_MAX_PENDENTES = max(1, _int("CURSO_MAX_PENDENTES", 4))

# ── Integração com o Trainify ──
# Chave enviada pelo Spring no cabeçalho X-Internal-Key.
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
# Sem chave, /api/* fica fechado por padrão. Desligar apenas em ambiente local.
REQUIRE_INTERNAL_KEY = os.getenv("REQUIRE_INTERNAL_KEY", "true").lower() == "true"
# Tamanho máximo de PDF baixado para ingestão.
MAX_PDF_BYTES = _int("MAX_PDF_MB", 50) * 1024 * 1024
MAX_PDF_PAGES = _int("MAX_PDF_PAGES", 500)
MAX_TEXT_CHARS = _int("MAX_TEXT_CHARS", 2_000_000)
MAX_REQUEST_BYTES = _int("MAX_REQUEST_MB", 72) * 1024 * 1024
# Hosts de onde o serviço aceita baixar PDFs (o Spring manda um link temporário
# do S3). Evita que o endpoint de ingestão seja usado para acessar a rede interna.
PDF_ALLOWED_HOSTS = [
    h.strip() for h in os.getenv("PDF_ALLOWED_HOSTS", "amazonaws.com,docs.google.com").split(",") if h.strip()
]

# ── Modo demonstração (interface web própria, disciplina de IA) ──
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
DEMO_COURSE_ID = os.getenv("DEMO_COURSE_ID", "demo-nrs")
DEMO_TENANT_ID = os.getenv("DEMO_TENANT_ID", "demo")
