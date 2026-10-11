# Proposta: gerar o curso com IA a partir dos documentos da empresa

> **Situação (11/10/2026): fases 1 a 3 implementadas e em produção.**
> Serviço de IA: `app/graphs/curso_graph.py`, `app/rascunhos.py`, endpoints `/api/course-drafts`.
> Trainify: `CourseGenerationService`, `CourseDraftController`, tabela `course_generation_drafts`
> (changelog 16) e a tela `/courses/ai` ("Criar com IA"). Diferenças em relação ao plano abaixo:
> a estrutura sai de **uma** chamada à LLM, a partir de um "mapa" do material montado sem LLM
> (em vez de resumir documento por documento), e os arquivos vão direto ao backend (multipart),
> sem passar pelo S3. O quiz por módulo continua pelo botão "Gerar com IA" na aula de avaliação
> criada automaticamente. Ainda não feito: "gerar de novo" uma aula específica (fase 4) e a fase 5.

> Documento para implementação. Escrito em 11/10/2026, depois da integração do
> Trainify com o serviço de IA (trainify-ai). Descreve o que já existe, o que
> falta e como construir a funcionalidade em fases.

## 1. O pedido

Uma empresa tem vários PDFs e artigos de treinamento. Na hora de criar um curso,
o instrutor sobe esses documentos e a IA **propõe o curso inteiro**: título,
descrição, módulos, aulas com conteúdo e, opcionalmente, um quiz por módulo.
O instrutor **revisa, edita e aprova**; só então o curso é criado (como rascunho)
no Trainify. Nada vai para os alunos sem a aprovação humana.

## 2. O que já existe e pode ser reaproveitado

### Repositórios

| Repositório | Caminho local | O que é |
|---|---|---|
| `Trainify-PI5/code` | `C:\dev\code-main` | Trainify: backend Spring Boot (`Backend/`), frontend React + Vite (`Frontend/`) |
| `luanbandeira/trainify-ai` | `C:\dev\microservico` | Serviço de IA em Python (FastAPI + LangGraph + FAISS), hospedado em `https://luanbandeira-trainify-ai.hf.space` |

### No serviço de IA (`trainify-ai`)

- `app/text_processing.py`: extração de texto de PDF (`pypdf`), limpeza (cabeçalho e rodapé repetidos, números de página, hifenização) e divisão em trechos (`dividir_em_chunks`, 900 caracteres com 150 de sobreposição).
- `app/embeddings.py`: modelo local `intfloat/multilingual-e5-small`, carregado uma única vez e com acesso serializado.
- `app/vector_store.py`: um índice FAISS por curso em `data/indexes/<course_id>/`; busca, listagem de trechos por aula ou módulo e remoção.
- `app/ingestao.py`: `indexar_aula(Aula, store, embedder)` aceita texto, transcrição e PDF.
- `app/graphs/quiz_graph.py`: grafo LangGraph que gera quiz a partir dos trechos de uma aula ou módulo, valida o JSON e tenta de novo quando vem inválido. **Pode ser reaproveitado por módulo do curso gerado.**
- `app/llm.py`: provedor de LLM configurável (`LLM_PROVIDER`/`LLM_MODEL`), tradução de 429 em `LLMLimiteAtingido`.
- Autenticação entre serviços: cabeçalho `X-Internal-Key` (`INTERNAL_API_KEY`).

### No Trainify

- `Backend/.../ai/TrainifyAiClient.java`: cliente Feign do serviço de IA (contratos como `record`).
- `Backend/.../ai/RagService.java`: tradução dos erros do serviço de IA para as exceções da API (`AiContentMissingException`, `AiUnavailableException`).
- `Backend/.../ai/AiCourseScope.java`: quem pode consultar o quê.
- `Backend/.../services/CourseService.java`: `createCourse`, `addModule`, `addLesson`. Ao salvar aulas, o índice da IA é atualizado sozinho (`AiIndexEvents`).
- `Backend/.../controllers/MediaController.java` + `S3Service`: upload de arquivos por URL pré-assinada (`/media/upload-url` e `/media/upload-complete`).
- `Frontend/src/pages/CourseBuilder.tsx`: editor de curso; `components/Course/AssessmentBuilder.tsx` já tem o padrão "Gerar com IA → revisar → salvar".

## 3. Fluxo proposto

```
Instrutor sobe os documentos (tela "Criar curso com IA")
        │
        ▼
Spring guarda os arquivos (S3) e cria um RASCUNHO DE GERAÇÃO (status: PROCESSANDO)
        │  envia os arquivos ao trainify-ai (trabalho assíncrono)
        ▼
trainify-ai
  1. extrai e limpa o texto de cada documento (já existe)
  2. indexa tudo num índice temporário "draft-<id>" (já existe)
  3. resume cada documento por partes (LLM, "map")
  4. propõe a estrutura: módulos e aulas, com as fontes de cada aula (LLM, "reduce")
  5. escreve o conteúdo de cada aula a partir dos trechos recuperados (RAG por aula)
  6. (opcional) gera um quiz por módulo (quiz_graph já existe)
        │  devolve um JSON com a proposta + fontes
        ▼
Spring salva a proposta no rascunho (status: PRONTO)
        │
        ▼
Tela de revisão: o instrutor edita títulos, junta/separa/remove módulos e aulas,
pede "gerar de novo" para uma aula, confere as fontes de cada aula
        │  "Aprovar"
        ▼
Spring cria o curso como DRAFT com CourseService (createCourse/addModule/addLesson)
e anexa os documentos originais. A indexação normal da IA acontece sozinha.
```

### Contrato sugerido (trainify-ai)

```
POST /api/course-drafts                 (X-Internal-Key)
  { "draft_id": "...", "tenant_id": "...", "language": "pt-BR",
    "documents": [ { "name": "Manual de EPI.pdf", "pdf_base64": "..." } | { "name": "...", "text": "..." } ],
    "target_modules": 3,               // opcional: sugestão do instrutor
    "with_quizzes": true }
  → 202 { "draft_id": "...", "status": "PROCESSING" }

GET /api/course-drafts/{draft_id}       (X-Internal-Key)
  → { "status": "PROCESSING" | "READY" | "FAILED", "progress": 0-100, "step": "Resumindo documento 2 de 5",
      "result": { "title": "...", "description": "...",
                  "modules": [ { "title": "...", "objective": "...",
                                 "lessons": [ { "title": "...", "content": "markdown",
                                                "sources": [ { "document": "Manual de EPI.pdf", "pages": "4-7" } ] } ],
                                 "quiz": { ...formato CreateAssessmentRequest... } } ] },
      "error": "..." }

POST /api/course-drafts/{draft_id}/lessons/regenerate   (refaz uma aula com instrução do instrutor)
DELETE /api/course-drafts/{draft_id}                    (descarta o índice temporário)
```

O trabalho deve rodar em segundo plano (thread ou fila simples) com o
progresso guardado em memória/disco. O Spring consulta o `GET` a cada poucos
segundos (polling) e repassa o progresso para a tela.

### Trainify (Spring + React)

- Tabela `course_generation_drafts` (migration nova): `id`, `tenant_id`, `created_by`, `status`, `progress`, `step`, `proposal_json` (JSONB), `document_ids`, `error`, `created_at`.
- Endpoints: `POST /ai/course-drafts` (multipart ou ids de media já enviada), `GET /ai/course-drafts/{id}`, `PUT /ai/course-drafts/{id}` (proposta editada), `POST /ai/course-drafts/{id}/approve` (cria o curso DRAFT), `DELETE`.
- Permissão: `SUPER_ADMIN`, `ADMIN`, `INSTRUCTOR`; isolamento por `tenant_id` como no resto do sistema.
- Frontend: página nova "Criar curso com IA" com três etapas: **Enviar documentos → Acompanhar geração → Revisar e aprovar**. A revisão reaproveita os componentes do `CourseBuilder`. Cada aula mostra "Fontes: Manual de EPI.pdf, p. 4-7".

## 4. Pontos de atenção

1. **Limite da LLM gratuita (o principal obstáculo).** A Groq gratuita permite ~8.000 tokens por minuto. Um curso a partir de 100 páginas faz dezenas de chamadas (resumos, estrutura, aulas, quizzes). Sem tratamento, estoura o limite em segundos. Opções: (a) fila que espaça as chamadas e respeita o `Retry-After`, mostrando progresso realista (alguns minutos); (b) plano pago da Groq (Dev Tier) durante a demonstração; (c) outro provedor configurável em `LLM_PROVIDER`.
2. **Tamanho do material.** Não cabe tudo num prompt. Por isso o desenho map → reduce: resumir por partes e só depois propor a estrutura. Limitar o MVP a ~5 documentos / ~150 páginas.
3. **Fidelidade.** O conteúdo de cada aula deve ser gerado só a partir dos trechos recuperados daquela aula (mesma regra do chat: não inventar), e cada aula guarda as fontes para o instrutor conferir.
4. **PDF escaneado.** Sem camada de texto, nada é extraído. Hoje o sistema só registra no log. Na geração de curso, avisar o instrutor ("o arquivo X parece ser uma imagem; envie uma versão com texto"). OCR fica para uma fase futura.
5. **Revisão obrigatória.** Nada é publicado automaticamente: a aprovação cria o curso como rascunho (`DRAFT`), e a publicação continua sendo um passo manual do instrutor.
6. **Regra da cadeira de IA.** A regra "LLM só na geração da resposta final" vale para o chatbot entregue na disciplina. Esta funcionalidade é do PI; usar a LLM em várias etapas aqui não afeta aquela entrega, mas o grafo do chat não deve ser alterado.

## 5. Fases

| Fase | Entrega | Tamanho estimado |
|---|---|---|
| 1 | Serviço de IA: `POST/GET /api/course-drafts` com extração, resumo, estrutura e conteúdo das aulas; testes com LLM falsa | médio |
| 2 | Trainify: tabela de rascunhos, endpoints, tela de envio e acompanhamento | médio |
| 3 | Tela de revisão com edição, fontes por aula e "aprovar" criando o curso DRAFT | médio |
| 4 | Quiz por módulo na proposta (reaproveita `quiz_graph`), "gerar de novo" por aula | pequeno |
| 5 | Melhorias: OCR, DOCX/PPTX, imagens | futuro |

## 6. Melhorias pequenas que valem antes (independentes desta proposta)

- **Mostrar ao instrutor se a aula foi indexada.** Hoje, se um PDF não tem texto, a falha fica só no log do servidor. O `GET /api/courses/{id}/lessons` do trainify-ai já informa quantos trechos cada aula tem; o editor de curso poderia mostrar "IA: pronta / sem texto aproveitável".
- **Recalibrar o filtro de relevância (`MIN_SCORE`, hoje 0,845) quando entrar material de outro assunto.** O valor foi medido com as NRs; o README do trainify-ai explica como medir.
