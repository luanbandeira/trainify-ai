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

# ── Quiz ──
QUIZ_MAX_CHUNKS = _int("QUIZ_MAX_CHUNKS", 12)
QUIZ_MAX_ATTEMPTS = _int("QUIZ_MAX_ATTEMPTS", 3)

# ── Integração com o Trainify ──
# Chave enviada pelo Spring no cabeçalho X-Internal-Key. Vazia = sem checagem
# (apenas para desenvolvimento local).
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
# Tamanho máximo de PDF baixado para ingestão.
MAX_PDF_BYTES = _int("MAX_PDF_MB", 50) * 1024 * 1024
# Hosts de onde o serviço aceita baixar PDFs (o Spring manda um link temporário
# do S3). Evita que o endpoint de ingestão seja usado para acessar a rede interna.
PDF_ALLOWED_HOSTS = [
    h.strip() for h in os.getenv("PDF_ALLOWED_HOSTS", "amazonaws.com,docs.google.com").split(",") if h.strip()
]

# ── Modo demonstração (interface web própria, disciplina de IA) ──
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
DEMO_COURSE_ID = os.getenv("DEMO_COURSE_ID", "demo-nrs")
DEMO_TENANT_ID = os.getenv("DEMO_TENANT_ID", "demo")
