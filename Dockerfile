# Imagem do trainify-ai (usada no Hugging Face Spaces e em qualquer hospedagem com Docker).
#
# O modelo de embeddings e o índice de demonstração das NRs são gerados no build,
# então o serviço sobe pronto, sem baixar nada. Os índices dos cursos do Trainify
# ficam em /app/data/indexes e são refeitos pelo próprio Trainify se o disco for
# apagado (ele confere periodicamente o que está indexado).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860 \
    REQUIRE_INTERNAL_KEY=true

# O Hugging Face Spaces roda o container com o usuário 1000
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/curso_nrs.json ./data/curso_nrs.json
COPY data/raw/manifest.json ./data/raw/manifest.json
COPY data/processed ./data/processed

# Baixa o modelo e monta o índice do curso de demonstração (leva ~2 min no build)
RUN python -m scripts.indexar_nrs && chown -R app:app /app

USER app
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
