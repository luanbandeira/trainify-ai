import pytest

from app import config
from app.text_processing import extrair_texto_pdf


@pytest.mark.parametrize("conteudo", [b"nao e pdf", b"%PDF-1.4\ncorrompido"])
def test_pdf_invalido_recebe_erro_compreensivel(conteudo):
    with pytest.raises(ValueError, match="PDF"):
        extrair_texto_pdf(conteudo)


def test_pdf_protegido_e_recusado():
    import io
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("senha")
    arquivo = io.BytesIO()
    writer.write(arquivo)
    with pytest.raises(ValueError, match="senha"):
        extrair_texto_pdf(arquivo.getvalue())


def test_pdf_respeita_limites_de_extracao(monkeypatch):
    from tests.test_api import _pdf_com_texto

    pdf = _pdf_com_texto("Conteudo do treinamento")
    monkeypatch.setattr(config, "MAX_TEXT_CHARS", 5)
    with pytest.raises(ValueError, match="texto"):
        extrair_texto_pdf(pdf)
    monkeypatch.setattr(config, "MAX_PDF_PAGES", 0)
    with pytest.raises(ValueError, match="páginas"):
        extrair_texto_pdf(pdf)


def test_limite_http_conta_blocos_sem_content_length(monkeypatch):
    import asyncio
    from app.request_limits import RequestSizeLimit

    monkeypatch.setattr(config, "MAX_REQUEST_BYTES", 5)
    mensagens = iter([
        {"type": "http.request", "body": b"123", "more_body": True},
        {"type": "http.request", "body": b"456", "more_body": False},
    ])
    respostas = []

    async def receive():
        return next(mensagens)

    async def send(mensagem):
        respostas.append(mensagem)

    async def app(scope, receive, send):
        pytest.fail("O corpo excedente não deve chegar à aplicação")

    asyncio.run(RequestSizeLimit(app)({"type": "http", "headers": []}, receive, send))
    assert respostas[0]["status"] == 413
