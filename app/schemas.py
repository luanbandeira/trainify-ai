"""Contratos da API (o que o Spring e a interface de demonstração enviam e recebem)."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from app import config

Papel = Literal["STUDENT", "INSTRUCTOR", "MANAGER", "ADMIN", "SUPER_ADMIN"]


class MensagemHistorico(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[MensagemHistorico] = Field(default_factory=list, max_length=50)
    role: Papel = "STUDENT"
    # Cursos que o usuário pode consultar. Quem decide é o Spring (matrículas,
    # cursos do instrutor, cursos da empresa); o serviço só obedece.
    course_ids: list[str] = Field(min_length=1, max_length=200)
    lesson_id: str | None = None


class Fonte(BaseModel):
    course_id: str | None
    module_id: str | None
    lesson_id: str | None
    lesson_title: str | None
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Fonte]


class QuizRequest(BaseModel):
    course_id: str
    lesson_id: str | None = None
    module_id: str | None = None
    num_questions: int = Field(default=5, ge=1, le=10)
    num_options: int = Field(default=4, ge=2, le=5)
    topic: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _escopo(self):
        if not self.lesson_id and not self.module_id:
            raise ValueError("informe lesson_id ou module_id")
        return self


class OpcaoQuiz(BaseModel):
    text: str
    isCorrect: bool


class PerguntaQuiz(BaseModel):
    text: str
    explanation: str
    options: list[OpcaoQuiz]


class QuizResponse(BaseModel):
    """Mesmo formato do CreateAssessmentRequest do Trainify (+ explicação e fontes)."""

    title: str
    passingScore: int
    maxAttempts: int
    questions: list[PerguntaQuiz]
    sources: list[str]


class IngestRequest(BaseModel):
    course_id: str
    title: str = Field(min_length=1, max_length=300)
    tenant_id: str | None = None
    module_id: str | None = None
    module_title: str | None = None
    content: str | None = Field(default=None, max_length=config.MAX_TEXT_CHARS)
    transcript: str | None = Field(default=None, max_length=config.MAX_TEXT_CHARS)
    pdf_url: str | None = None       # link temporário (pré-assinado) do S3
    pdf_base64: str | None = None    # alternativa: o próprio arquivo


class IngestResponse(BaseModel):
    lesson_id: str
    course_id: str
    chunks: int


class DemoChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[MensagemHistorico] = Field(default_factory=list, max_length=50)
    role: Papel = "STUDENT"
    lesson_id: str | None = None


class DemoQuizRequest(BaseModel):
    lesson_id: str | None = None
    module_id: str | None = None
    num_questions: int = Field(default=3, ge=1, le=10)
    num_options: int = Field(default=4, ge=2, le=5)
    topic: str | None = Field(default=None, max_length=300)


class DocumentoCurso(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    pdf_base64: str | None = None
    text: str | None = Field(default=None, max_length=config.MAX_TEXT_CHARS)

    @model_validator(mode="after")
    def _conteudo(self):
        if self.pdf_base64 and self.text:
            raise ValueError("envie pdf_base64 ou text, não ambos")
        if not self.pdf_base64 and not (self.text and self.text.strip()):
            raise ValueError(f'o documento "{self.name}" não tem conteúdo (envie pdf_base64 ou text)')
        return self


class RascunhoCursoRequest(BaseModel):
    draft_id: str
    documents: list[DocumentoCurso] = Field(min_length=1, max_length=config.CURSO_MAX_DOCUMENTOS)
    min_modules: int = Field(default=2, ge=1, le=8)
    max_modules: int = Field(default=5, ge=1, le=8)
    instructions: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _faixa(self):
        if self.min_modules > self.max_modules:
            raise ValueError("min_modules não pode ser maior que max_modules")
        nomes = [d.name.lower() for d in self.documents]
        if len(set(nomes)) != len(nomes):
            raise ValueError("há documentos com o mesmo nome")
        return self
