# Trainify AI — RAG para treinamento corporativo

Microserviço em Python que dá inteligência artificial ao **Trainify** (plataforma
de treinamento corporativo, projeto de PI5). Ele resolve dois problemas:

1. **Chat com o conteúdo dos cursos:** o aluno, o instrutor e o gestor fazem perguntas e
   recebem respostas baseadas **apenas** no material dos cursos que podem ver, com as fontes.
2. **Geração de quiz:** a partir do material que o instrutor subiu (texto, PDF ou
   transcrição), a IA sugere perguntas de múltipla escolha, que o instrutor revisa antes de publicar.

O serviço funciona sozinho, com interface web própria e uma base de conhecimento
sobre as **Normas Regulamentadoras (NRs)** de segurança do trabalho. Também se liga
ao backend Spring Boot do Trainify por HTTP.

```mermaid
flowchart LR
    subgraph Offline[Montagem da base]
        A[gov.br<br/>15 NRs em PDF] -->|coleta| B[data/raw]
        B -->|limpeza| C[data/processed]
        C -->|chunking<br/>900 caracteres| D[chunks]
        D -->|embeddings<br/>multilingual-e5-small| E[(FAISS<br/>um índice por curso)]
    end
    subgraph Chat[Grafo LangGraph de chat]
        direction LR
        N1[receber_pergunta] --> N2[montar_consulta] --> N3[recuperar_contexto]
        N3 -->|trechos relevantes| N4[montar_prompt<br/>chunks + histórico] --> N5[chamar_llm] --> N6[formatar_resposta]
        N3 -->|nada relevante| N7[sem_contexto] --> N6
    end
    E -.-> N3
    UI[Interface web / Trainify] -->|pergunta + histórico| N1
    N6 -->|resposta + fontes| UI
```

## Atendimento aos requisitos da atividade

| Requisito | Onde está |
|---|---|
| Interface web simples (pergunta, envio, resposta, comunicação com o back-end) | `app/static/index.html`, servida em `http://localhost:8000` |
| Back-end em Python | FastAPI, `app/main.py` |
| RAG orquestrado com LangGraph, com etapas explícitas | `app/graphs/chat_graph.py`: `receber_pergunta → montar_consulta → recuperar_contexto → montar_prompt → chamar_llm → formatar_resposta` |
| Base de conhecimento de domínio específico e volumosa | 15 NRs: **447 páginas de PDF, ~146 mil palavras, 1.327 chunks** (detalhes abaixo) |
| LLM externa usada só na geração | Só o nó `chamar_llm` chama a LLM. Embeddings, busca e perguntas de seguimento rodam localmente, sem LLM |
| Contexto com os chunks recuperados + mensagens anteriores | Nó `montar_prompt`: trechos no prompt de sistema e as últimas 6 mensagens como histórico |
| Sem API pronta de chatbot | A API (`/demo/chat`, `/api/chat`) é nossa; a LLM é usada como modelo de linguagem via LangChain |

## Base de conhecimento

- **Domínio:** segurança e saúde no trabalho, Normas Regulamentadoras do Ministério do Trabalho e Emprego.
- **Fonte:** textos oficiais consolidados, publicados em [gov.br/trabalho-e-emprego](https://www.gov.br/trabalho-e-emprego/pt-br/acesso-a-informacao/participacao-social/conselhos-e-orgaos-colegiados/comissao-tripartite-partitaria-permanente/normas-regulamentadora/normas-regulamentadoras-vigentes) (domínio público).
- **Volume:** 15 normas, 447 páginas, ~146 mil palavras, 1.327 chunks indexados.
- **Organização:** um curso de demonstração com 3 módulos, em que cada NR é uma aula (`data/curso_nrs.json`):

| Módulo | Aulas |
|---|---|
| Gestão de segurança e saúde no trabalho | NR-1, NR-5, NR-7, NR-9 |
| Proteção individual e ambiente de trabalho | NR-6, NR-17, NR-23, NR-24, NR-26 |
| Atividades de alto risco | NR-10, NR-11, NR-12, NR-18, NR-33, NR-35 |

- **Origem de cada arquivo** (URL e data da coleta): `data/raw/manifest.json`
- **Textos limpos:** `data/processed/*.txt`, versionados. Não é preciso rodar a coleta de novo para executar o projeto.

## Pipeline

| Etapa | Arquivo | O que faz |
|---|---|---|
| 1. Coleta | `scripts/coletar_nrs.py` | Acha na página do gov.br o PDF consolidado mais recente de cada NR e baixa para `data/raw/` |
| 2. Limpeza | `scripts/processar_nrs.py` + `app/text_processing.py` | Extrai o texto (pypdf), tira cabeçalhos e rodapés repetidos, números de página, histórico de portarias e sumário, desfaz a hifenização e refaz os parágrafos (cada item numerado da norma vira um parágrafo) |
| 3. Chunking | `app/text_processing.py` | `RecursiveCharacterTextSplitter`, 900 caracteres com 150 de sobreposição, quebrando de preferência entre parágrafos. Cada chunk recebe o título da aula como prefixo |
| 4. Embeddings | `app/embeddings.py` | `intfloat/multilingual-e5-small`, local e multilíngue, com os prefixos `passage:` e `query:` |
| 5. Índice vetorial | `app/vector_store.py` | FAISS `IndexFlatIP` sobre vetores normalizados (similaridade de cosseno), **um índice por curso**, persistido em `data/indexes/<curso>/` |
| 6–10. Pergunta → resposta | `app/graphs/chat_graph.py` | O grafo LangGraph descrito abaixo |

### Grafo de chat

1. **`receber_pergunta`**: normaliza a pergunta e corta o histórico nas últimas `HISTORY_MESSAGES` mensagens.
2. **`montar_consulta`**: trata perguntas de seguimento **sem LLM**. Se a pergunta é curta ou começa com "e…", "mas…", "isso…", a busca junta a pergunta anterior do usuário com a atual. Assim "e acima de quantos metros?" é buscada junto com "quando a NR de trabalho em altura se aplica?".
3. **`recuperar_contexto`**: busca os trechos mais próximos **apenas nos cursos permitidos** (`course_ids`). Se o usuário está numa aula, os trechos dela ganham prioridade no desempate. Só passam os trechos com similaridade ≥ `MIN_SCORE`.
4. **`montar_prompt`**: monta o prompt de sistema (regras gerais + regras do **perfil**: aluno, instrutor ou gestor) com os trechos numerados e a aula de origem de cada um, depois acrescenta o histórico e a pergunta.
5. **`chamar_llm`**: **a única chamada à LLM externa.**
6. **`formatar_resposta`**: devolve a resposta e as fontes (aulas usadas, com a nota de similaridade).

Se nenhum trecho passa no filtro, o grafo vai para **`sem_contexto`** e responde que o material não cobre o assunto. Nesse caso a LLM não é chamada, porque sem contexto ela só poderia inventar.

### Grafo de quiz (`app/graphs/quiz_graph.py`)

`carregar_material → selecionar_trechos → montar_prompt → chamar_llm → validar_quiz`, com volta para
`montar_prompt` quando a validação falha (até `QUIZ_MAX_ATTEMPTS` vezes). Nesse caso os erros encontrados vão junto no novo prompt.

- **`selecionar_trechos`**: sem foco, pega trechos espalhados pelo material inteiro, para cobrir todos os assuntos. Com foco (ex.: "análise de risco"), pega os trechos mais próximos do tema.
- **`validar_quiz`**: confere o JSON, o número de perguntas e de alternativas, que haja exatamente uma correta, que não haja alternativas repetidas nem "todas as anteriores". Depois embaralha as alternativas, porque LLMs tendem a pôr a correta sempre em primeiro.
- A saída segue o formato do `CreateAssessmentRequest` do Trainify, mais uma explicação por pergunta.

### Perfis

| Perfil (`role`) | Comportamento |
|---|---|
| `STUDENT` | Tutor: explica para o aluno aprender, de forma breve, e **nunca entrega gabarito** |
| `INSTRUCTOR` | Também ajuda no trabalho pedagógico: sugere questões (com a correta e o trecho que a justifica), resumos, objetivos e pontos a reforçar |
| `MANAGER`, `ADMIN`, `SUPER_ADMIN` | Foco em aplicação: obrigações, responsabilidades, quem precisa de qual treinamento. Respostas em tópicos |

O que cada perfil **pode ver** não é decidido aqui: o backend do Trainify manda em `course_ids` só os cursos permitidos para aquele usuário.

## Como executar

### Pré-requisitos

- **Python 3.12** (`py -3.12 --version` no Windows). O Python 3.13 não tem pacote do `faiss-cpu` 1.9.
- **Chave da Groq**, gratuita: crie em https://console.groq.com/keys.
- ~2 GB livres (dependências + modelo de embeddings) e internet na primeira execução.

### Passo a passo

Na raiz do projeto:

```bash
# 1. ambiente virtual
py -3.12 -m venv venv                  # Windows  (Linux/macOS: python3.12 -m venv venv)
.\venv\Scripts\Activate.ps1            # Windows  (Linux/macOS: source venv/bin/activate)

# 2. dependências (torch CPU primeiro, para não baixar pacotes CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. configuração: copie o exemplo e cole sua chave em GROQ_API_KEY
copy .env.example .env                 # Linux/macOS: cp .env.example .env

# 4. índice vetorial (obrigatório na 1ª vez; baixa o modelo de embeddings, ~470 MB, e leva ~2 min)
python -m scripts.indexar_nrs

# 5. API + interface
uvicorn app.main:app --port 8000
```

Abra **http://localhost:8000**. Na aba **Chat**, escolha o perfil e pergunte. Na aba **Gerar quiz**, escolha uma aula ou um módulo.
A documentação interativa da API fica em http://localhost:8000/docs.

> Se o PowerShell disser que "a execução de scripts foi desabilitada", rode
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` e ative o ambiente de novo.

### Refazer a base do zero (opcional)

```bash
python -m scripts.coletar_nrs      # baixa os PDFs do gov.br para data/raw/
python -m scripts.processar_nrs    # limpa e grava data/processed/
python -m scripts.indexar_nrs      # chunking + embeddings + FAISS
```

### Testes

Rodam offline: a LLM e o modelo de embeddings são substituídos por dublês, então não precisam de chave nem de índice.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Integração com o Trainify

O navegador nunca fala direto com este serviço: o backend Spring Boot é o portão.
Ele autentica o usuário, descobre a empresa e o perfil, decide quais cursos o usuário
pode consultar e só então chama o `trainify-ai`, enviando o cabeçalho `X-Internal-Key`.

```
React ──JWT──▶ Spring Boot ──HTTP + X-Internal-Key──▶ trainify-ai
```

| Endpoint | Quando o Spring chama | Corpo |
|---|---|---|
| `POST /api/ingest/lessons/{lessonId}` | Aula criada ou editada (de forma assíncrona) | `course_id, title, tenant_id, module_id, module_title, content, transcript` e `pdf_url` (link pré-assinado do S3) **ou** `pdf_base64` |
| `DELETE /api/ingest/courses/{courseId}/lessons/{lessonId}` | Aula excluída | – |
| `GET /api/courses/{courseId}/lessons` | Saber o que já está indexado | – |
| `POST /api/chat` | Mensagem no chat do Trainify | `question, history[], role, course_ids[], lesson_id?` |
| `POST /api/quiz` | Instrutor clica em "Gerar quiz com IA" | `course_id, lesson_id` ou `module_id, num_questions, num_options, topic?` |
| `GET /health` | Mostrar ou esconder os recursos de IA na tela | – |

Respostas de erro: `401` chave interna errada, `422` entrada inválida ou aula sem material,
`429` limite de uso da LLM atingido (aguardar alguns segundos), `502` a LLM não gerou um quiz válido,
`503` a LLM está indisponível.

> **Limite do plano gratuito da Groq:** cerca de 8.000 tokens por minuto. Cada pergunta usa
> ~2.500 tokens (trechos + histórico), então perguntas muito seguidas podem receber 429.
> O serviço já tenta de novo respeitando o tempo de espera que a Groq indica.

### Segurança

- **Isolamento entre empresas:** um índice FAISS por curso, e a busca só abre os cursos de `course_ids`.
- **Ids viram nomes de pasta:** só são aceitos `[A-Za-z0-9_-]`, o que impede caminhos como `../`.
- **Download de PDF** só por HTTPS e só dos hosts de `PDF_ALLOWED_HOSTS` (S3 por padrão), com limite de tamanho, para que a ingestão não possa ser usada para acessar a rede interna.
- **Modo de demonstração** (`DEMO_MODE`): os endpoints `/demo/*` só enxergam o curso das NRs. Em produção, use `DEMO_MODE=false`.

## Configuração

Todas as variáveis estão em `.env.example`. As principais:

| Variável | Padrão | Para quê |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | `groq` / `openai/gpt-oss-20b` | Provedor e modelo do chat. Troca sem mudar código (`openai`, `anthropic`, `google_genai`… instalando o pacote `langchain-*` correspondente) |
| `QUIZ_LLM_MODEL` | `openai/gpt-oss-120b` | Modelo usado só no quiz. Nos testes, o 20b gerava alternativas "erradas" que a norma considera certas |
| `LLM_MAX_RETRIES` | 4 | Novas tentativas quando o provedor recusa por limite (HTTP 429) |
| `TOP_K` | 6 | Quantos trechos vão para o prompt |
| `MIN_SCORE` | 0.845 | Similaridade mínima para um trecho ser usado |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 900 / 150 | Tamanho e sobreposição dos chunks |
| `HISTORY_MESSAGES` | 6 | Mensagens anteriores levadas ao prompt |
| `INTERNAL_API_KEY` | vazio | Chave que o Spring envia em `X-Internal-Key` |

### Calibração do `MIN_SCORE`

O e5 dá notas altas até para textos sem relação. Por isso o corte foi medido na própria base:

| Pergunta | Melhor nota | Resultado |
|---|---|---|
| "A partir de qual altura a NR-35 se aplica?" | 0,909 (NR-35) | usa |
| "pausas para quem trabalha com computador" | 0,891 (NR-17) | usa |
| "o que fazer se o extintor estiver vencido" | 0,853 | usa |
| "meu chefe pode me obrigar a trabalhar sem luva?" | 0,847 | usa |
| "Como configurar o Kubernetes?" | 0,841 | descarta |
| "Receita de bolo de chocolate" | 0,838 | descarta |
| "melhor time de futebol do Brasil" | 0,818 | descarta |
| "Qual a capital da França?" | 0,769 | descarta |

## Estrutura

```
app/
  main.py              API FastAPI (rotas /api, /demo, /health)
  config.py            configuração via .env
  text_processing.py   extração de PDF, limpeza e chunking
  embeddings.py        modelo de embeddings local
  vector_store.py      índices FAISS por curso
  ingestao.py          ingestão de uma aula (texto, PDF, transcrição)
  llm.py               LLM externa (provedor configurável)
  prompts.py           instruções por perfil e do quiz
  graphs/chat_graph.py grafo de RAG do chat
  graphs/quiz_graph.py grafo de geração de quiz
  static/index.html    interface web
scripts/               coleta, limpeza e indexação da base de NRs
data/
  curso_nrs.json       estrutura do curso de demonstração
  raw/manifest.json    origem de cada PDF
  processed/           textos limpos
tests/                 testes offline (pytest)
```

## Dependências

Listadas em `requirements.txt`, com versões fixas: FastAPI, Uvicorn, LangGraph, LangChain (+ `langchain-groq`),
sentence-transformers, faiss-cpu, NumPy, pypdf, httpx e python-dotenv. Os testes também usam pytest (`requirements-dev.txt`).
