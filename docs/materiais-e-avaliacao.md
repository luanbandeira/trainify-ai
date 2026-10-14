# Materiais atualizados e respostas verificáveis

## Situação dos materiais

`GET /api/courses/{course_id}/status` exige a chave interna e devolve a situação do curso e de cada aula conhecida. O contrato de ingestão existente continua válido.

| Situação da aula | Significado |
| --- | --- |
| READY | Material disponível na versão informada |
| PROCESSING | Atualização em andamento neste processo |
| EMPTY | O último envio não contém texto aproveitável |
| FAILED | A última atualização falhou |
| INTERRUPTED | O índice foi reaberto com atualização pendente de uma execução anterior |

Um curso desconhecido retorna MISSING. Um curso sem aulas conhecidas retorna EMPTY. Um curso com aulas indisponíveis retorna UNAVAILABLE; os demais retornam READY. Não se presume que um curso sem índice esteja sendo processado.

O estado é gravado no snapshot antes de processar uma atualização. Enquanto a aula estiver indisponível, os trechos anteriores não entram no chat nem no quiz. Outras aulas disponíveis continuam consultáveis, e o chat avisa quando o escopo está incompleto. Reenviar o material com sucesso devolve a aula a READY. Excluir a aula remove seu conteúdo e estado.

Os estados e as versões persistem em disco, mas dependem de armazenamento durável na hospedagem. A operação continua restrita a um processo; múltiplos processos exigem coordenação compartilhada. Uma falha do próprio disco pode impedir o registro da situação e precisa de tratamento operacional.

O serviço só conhece alterações recebidas pela API. O backend deve enviar edições e exclusões e pode consultar o novo endpoint para apresentar os estados na plataforma. Não é possível detectar uma edição que nunca foi comunicada.

## Referências

Cada trecho tem `source_id` e `version`. A versão é uma impressão SHA-256 do conteúdo e dos metadados recebidos, não uma data de vigência ou numeração editorial. Reenvios idênticos preservam a identidade. Índices legados podem retornar versão nula até a próxima ingestão.

As fontes preservam trechos distintos da mesma aula. O campo `kind: retrieved` identifica material consultado pelo modelo: não constitui comprovação automática de que todas as afirmações da resposta estão sustentadas. A interface permite conferir cada trecho e deixa essa distinção explícita. Os campos antigos permanecem disponíveis para compatibilidade com o Spring.

## Avaliar respostas finais

A base `data/avaliacao/respostas_corporativas.json` contém 14 cenários de empresas fictícias, incluindo acesso, incidentes, visitantes, mudança de assunto, perfis e isolamento. Ela complementa a avaliação de recuperação existente. Não é evidência de precisão em produção nem uma base exaustiva.

Com chave de LLM e modelo de embeddings configurados:

```bash
python -m scripts.avaliar_respostas --gerar --saida avaliacao-respostas.json
```

Essa ação consome a cota do provedor. Os materiais são indexados numa pasta temporária, sem alterar os índices de produção. O arquivo guarda respostas, fontes, configuração, fatos esperados e campos de revisão inicialmente vazios. Falhas ficam registradas por tipo, sem conteúdo de exceções do provedor.

Uma pessoa deve preencher `reviewer`, `notes` e os cinco campos booleanos de `review`, comparando a resposta com os materiais:

- `correta`: não contradiz os fatos esperados e não inventa regras.
- `fundamentada`: as afirmações podem ser justificadas pelo material autorizado.
- `completa`: cobre o necessário para responder à pergunta, incluindo uma recusa adequada quando cabível.
- `fontes_sustentam`: os trechos apresentados permitem verificar o que foi afirmado; uma recusa sem afirmações factuais pode ser aprovada sem fontes.
- `perfil_adequado`: linguagem e orientação adequadas ao papel solicitado.

```bash
python -m scripts.avaliar_respostas --resumir avaliacao-respostas.json
```

O resumo conta somente revisões completas e identificadas. Sem revisão humana, as medidas de qualidade ficam nulas, nunca 100%. Os checks automáticos de escopo, presença de aulas esperadas e marcador de recusa são diagnósticos, não avaliação semântica da resposta. Uma recusa redigida de outra forma requer revisão humana.

Para comparar versões, use a mesma base, registre configuração e repita execuções. Mantenha casos adicionais reservados para validação, sem usá-los para ajustar prompts e limiares. Não confunda os testes offline, que usam dublês, com uma avaliação executada no modelo real.
